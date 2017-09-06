//
//  main.cpp
//  ncnn_mobileNet
//
//  Created by Wenbo Huang on 2017/9/6.
//  Copyright © 2017年 Wenbo Huang. All rights reserved.


#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/time.h>
#include <unistd.h>

#include "net.h"
using namespace std;
using namespace cv;

long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

static int detect_mobileNet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net mobileNet;
    mobileNet.load_param("/Users/Guigu/Documents/projects/ncnn_mobileNet/mobilenet.param");
    mobileNet.load_model("/Users/Guigu/Documents/projects/ncnn_mobileNet/mobilenet.bin");
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    
    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f,0.017f,0.017f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    
    ncnn::Extractor ex = mobileNet.create_extractor();
    ex.set_light_mode(true);
    
    ex.input("data", in);
    
    ncnn::Mat out;
    ex.extract("fc7", out);
    
    cls_scores.resize(out.c);
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }
    
    return 0;
}

static int load_labels(string path, std::vector<string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");
    
    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);  //¶ÁÈ¡Ò»ÐÐ
        string str_s(str);
        
        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    vector<string> labels;
    load_labels("/Users/Guigu/Documents/projects/ncnn/synset_words.txt", labels);
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());
    
    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        //fprintf(stderr, " %s", labels[index].c_str());
        fprintf(stderr, "%d = %f (%s)\n", index, score, labels[index].c_str());
    }
    
    return 0;
}

int main(int argc, char** argv)
{
    
    const char* imagepath = "/Users/Guigu/Downloads/lufei.jpg";
    
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    
    std::vector<float> cls_scores;
    
    long time = getTimeUsec();
    detect_mobileNet(m, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ldms\n",time/1000);
    
    print_topk(cls_scores, 3);
    
    return 0;
}

