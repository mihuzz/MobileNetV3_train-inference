#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/video.hpp>
#include <ATen/ATen.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torchvision/vision.h>

#define kImage_size 224
#define back_size_w 640
#define back_size_h 480
#define kChannels 3

const float scale = 0.003921569; // 1/255=0.003921569
const float scoreThreshold = 0.7;



int main()
{
    std::vector<std::string> classes;
    const std::string videofile;

    std::string file = "/home/mihuzz/QTprojects/c10test4jit/classes.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }


    std::ifstream is("/home/mihuzz/Видео/JIT_faster_mobileV3_large_CUDA.pt");

    torch::jit::script::Module network = torch::jit::load(is, torch::kCUDA);

    network.to(at::kCUDA);
    network.eval();

    double t_tick = 0;
    double fps = 0;

    cv::VideoCapture cap;
    cv::VideoWriter myvideo("/home/mihuzz/Видео/outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),60, cv::Size(640,480));

        if(videofile != "")
            cap.open(videofile);

        else
            cap.open(0, cv::CAP_V4L2);
        cv::Mat frame;

        int count = 0;

        while (cv::waitKey(1)<0)
        {
//            cv::Mat frame;
            cap >> frame;
//            cv::cuda::GpuMat gpu_frame;
//            gpu_frame.upload(frame);
            ++ count;

            double t_start = cv::getTickCount();

            if (frame.empty())
            {
                cv::waitKey();
                break;
            }

            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            cv::Size scaleFrame(kImage_size, kImage_size);
            cv::resize(frame,frame, scaleFrame);

            frame.convertTo(frame, CV_32FC3, scale); // convert for tensor data type input

            torch::TensorOptions options(torch::kFloat);

            auto input_tensor = torch::from_blob(frame.data,{kImage_size,kImage_size, kChannels}, options);
            input_tensor = input_tensor.permute({2,0,1}).contiguous();
            input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
            input_tensor[0][1] = input_tensor[0][0].sub_(0.456).div_(0.224);
            input_tensor[0][2] = input_tensor[0][0].sub_(0.406).div_(0.225);

            input_tensor = input_tensor.to(at::kCUDA);
//
            std::vector<torch::Tensor> images;
            images.emplace_back(input_tensor);

            auto outputs = network.forward({images});

            auto val_dict = outputs.toTuple()->elements().at(1).toList().get(0).toGenericDict();

            auto boxes = val_dict.at("boxes").toTensor().to(torch::kCPU).to(torch::kInt16);
            auto scores = val_dict.at("scores").toTensor().to(torch::kCPU).to(torch::kF16);
            auto labels = val_dict.at("labels").toTensor().to(torch::kCPU).to(torch::kInt8);


            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            cv::Size scaleFrame2(back_size_w, back_size_h);
            cv::resize(frame,frame, scaleFrame2);            
            frame.convertTo(frame, CV_8UC3, 255);

            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<cv::Rect> Rectboxes;

            if (scores.numel())
            {
                    for (int i = 0; i < scores.numel(); ++i)
                    {
                        if (scores[i].item<float>() > scoreThreshold)
                        {
                            auto x1 = static_cast<int>((boxes[i][0].item<float>()/224.f)*640.f);
                            auto y1 = static_cast<int>((boxes[i][1].item<float>()/224.f)*480.f);
                            auto x2 = static_cast<int>((boxes[i][2].item<float>()/224.f)*640.f);
                            auto y2 = static_cast<int>((boxes[i][3].item<float>()/224.f)*480.f);

                            auto conf = scores[i].item<float>();
                            auto labelID = labels[i].item<int>();

                            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
                            std::string info_on_image = cv::format("%s : %.3f", (classes.empty() ?
                                      cv::format(" #%d", labelID).c_str() : classes[labelID].c_str()), conf);
                            cv::putText(frame, info_on_image, cv::Point(x1, y1-5),
                                        cv::FONT_HERSHEY_COMPLEX , 0.7, cv::Scalar(0, 255, 0), 2, 5);
      ;
                        }
                    }
            }

            t_tick = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();

            fps = 1/t_tick;

            cv::putText(frame, cv::format("fps :%.3f", fps), cv::Point(10, 20),
                        cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0,255,0), 2, 5);

//            cv::String name  = "faster_mobile_v3_frame" + std::to_string(static_cast<long>(count)) + ".jpg";
            cv::imshow("faster_mobile_v3_frame", frame);
//            cv::imwrite("/home/mihuzz/Видео/" + name, frame);
//            myvideo.write(frame);
        }
    return 0;
}
