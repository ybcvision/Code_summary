# 代码模块总结
***

## 目录
[TOC]

## 一、C++

### 1、opencv
____
#### 1.1、一个透明度的PNGLogo加入到一张大图的指定位置

注意：类似于copyto的操作（to不能处理透明的图片）。

```
//定义和读取
        cv::Mat                     image_;                     /*! <相机图像 */
		cv::Mat                     Title_image;                /*! <标题图片 */
		
		/**
		  * @brief  将LOGO插入到图片中
		  * @date   2018年9月20日
		  * @param[in]  int rowStart  图像行
		  * @param[in]  int colStart  图像列
		  */
		void InsertLogo(Mat image, Mat logoImage, int rowStart, int colStart);
		
//读取透明度图片（4通道）时用CV_LOAD_IMAGE_UNCHANGED
Title_image = cv::imread("./inform_image/1.png", CV_LOAD_IMAGE_UNCHANGED);

//核心函数,其中rowStart为插入行位置、colStart为插入列位置;
void Camera::InsertLogo(Mat image, Mat logoImage, int rowStart , int colStart)
	{
		image_ = image;

		for (int i = rowStart; i < logoImage.rows+ rowStart && i < image_.rows; i++)
			for (int j = colStart; j < logoImage.cols + colStart && j < image_.cols; j++)
			{

				float ratio = float(logoImage.at<Vec4b>(i-rowStart, j-colStart)[3]) / 255.f;
				for (int ii = 0; ii < 3; ii++)
				{
					image_.at<Vec4b>(i, j)[ii] = uchar(float(logoImage.at<Vec4b>(i- rowStart, j- colStart)[ii]) * ratio
						+ float(image_.at<Vec4b>(i, j)[ii]) * (1.f - ratio));
				}
			}
	}

//主代码

		cv::cvtColor(image_, image_, cv::COLOR_BGR2BGRA);//先将RGB的三通道图转变为四通道
		InsertLogo(image_, Title_image, 20, 800);//调用函数

```
____
#### 1.2、putText加入中文内容

注：为了解决opencv中putText不能打中文

____




### 2、tensorflow

____
#### 2.1、Fast_Rcnn物体检测
注：这是用C++ 使用经过python训练出的model进行物体检测，并且进行输出的代码

1、头函数
DeeplearningDetector.h
```
#pragma once
#include <vector>
#include "Saveditem.h"
#include <opencv2/core/core.hpp>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace std;
namespace visionnav
{
	namespace classification
	{
		class DeeplearningDetector
		{
		public:
			DeeplearningDetector();
			~DeeplearningDetector();

			void init();
	
			void close();
	
			vector<Saveditem> DL_Detector(cv::Mat &img);
			vector<Saveditem> itemInfomation;
	
		};
	}
}


```

utils.h
```
#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/core.hpp>

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;

Status readLabelsMapFile(const string &fileName, std::map<int, string> &labelsMap);

Status loadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);

Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);

void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score, std::string label, bool scaled);

void drawBoundingBoxesOnImage(cv::Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              std::map<int, string> &labelsMap,
                              std::vector<size_t> &idxs);

double IOU(cv::Rect box1, cv::Rect box2);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                                tensorflow::TTypes<float, 3>::Tensor &boxes,
                                double thresholdIOU, double thresholdScore);

#endif //TF_DETECTOR_EXAMPLE_UTILS_H
```

源代码
DeeplearningDetector.cpp

```
#pragma once
#include "DeeplearningDetector.h"
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include "Timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>

#include "utils.h"
#include "Saveditem.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

namespace visionnav
{
	namespace classification 
	{
		DeeplearningDetector::DeeplearningDetector() {};
		DeeplearningDetector::~DeeplearningDetector() {};
		// Load labels map from .pbtxt file
		std::map<int, std::string> labelsMap = std::map<int, std::string>();
		// Load and initialize the model from .pb file
		std::unique_ptr<tensorflow::Session> session;

		void DeeplearningDetector::init()
		{
			// Set dirs variables
			string ROOTDIR = "./";
			string LABELS = "model/labelmap.pbtxt";
			string GRAPH = "model/frozen_inference_graph.pb";
	
			// 设置输出;
			vector<Saveditem> itemInfomation;
	
			string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
			LOG(INFO) << "graphPath:" << graphPath;
			Status loadGraphStatus = loadGraph(graphPath, &session);
			if (!loadGraphStatus.ok()) {
				LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
			}
			else
			{
				LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;
			}
				
			Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR,LABELS), labelsMap);
			if (!readLabelsMapStatus.ok()) {
				LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
			}
			else
				LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;
		}
	
		vector<Saveditem> DeeplearningDetector::DL_Detector(cv::Mat &img)
		{
			// Set input & output nodes names
			string inputLayer = "image_tensor:0";
			vector<string> outputLayer = { "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
			itemInfomation.clear();
	
			//输入图像;
			Mat frame = img;
			resize(frame, frame, Size(frame.cols/3, frame.rows/3));
			Tensor tensor;
			std::vector<Tensor> outputs;
			double thresholdScore = 0.95;
			double thresholdIOU = 100;
			tensorflow::TensorShape shape = tensorflow::TensorShape();
			shape.AddDim(1);;
			shape.AddDim(frame.rows);
			shape.AddDim(frame.cols);
			shape.AddDim(3);
			tensor = Tensor(tensorflow::DT_FLOAT, shape);
			Status readTensorStatus = readTensorFromMat(frame, tensor);
			if (!readTensorStatus.ok())
			{
				LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
			}
	
			// Run the graph on tensor
			outputs.clear();
	
			Status runStatus = session->Run({ { inputLayer, tensor } }, outputLayer, {}, &outputs);
			if (!runStatus.ok()) {
				LOG(ERROR) << "Running model failed: " << runStatus;
			}
	
			// Extract results from the outputs vector
			tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
			tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
			tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
			tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();
	
			vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
	
			// Draw bboxes and captions
			drawBoundingBoxesOnImage(img, scores, classes, boxes, labelsMap, goodIdxs);
			for (size_t i = 0; i < goodIdxs.size(); i++)
			{
				//LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
				//	<< " (" << classes(goodIdxs.at(i)) << "), box:" << boxes(0, goodIdxs.at(i), 0) << ","
				//	<< boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
				//	<< boxes(0, goodIdxs.at(i), 3);
				string _itemclass = labelsMap[classes(goodIdxs.at(i))];
				double _itemX1 = img.cols* boxes(0, goodIdxs.at(i), 1);
				double _itemY1 = img.rows* boxes(0, goodIdxs.at(i), 0);
				double _itemX2 = img.cols* boxes(0, goodIdxs.at(i), 3);
				double _itemY2 = img.rows* boxes(0, goodIdxs.at(i), 2);
				itemInfomation.push_back(Saveditem(img, _itemclass, _itemX1, _itemX2, _itemY1, _itemY2));
			}
			return itemInfomation;
		}
	
		void DeeplearningDetector::close()
		{
			itemInfomation.clear();
			session->Close();
			session->~Session();
			waitKey(50);
			DeeplearningDetector::~DeeplearningDetector();
		}
	
	}
}

```
utils.cpp（功能模块）
```
#include "utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>
#include <numeric>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

typedef Rect_ <float> Rect2f;

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

/** Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
 */
Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();
    
    // Search entry patterns of type 'item { ... }' and parse each of them
    smatch matcherEntry;
    smatch matcherId;
    smatch matcherName;
    const regex reEntry("item \\{([\\S\\s]*?)\\}");
    const regex reId("[0-9]+");
    const regex reName("\'.+\'");
    string entry;
    
    auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = sregex_iterator();
    
    int id;
    string name;
    for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
        matcherEntry = *i;
        entry = matcherEntry.str();
        regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = stoi(matcherId[0].str());
        else
            continue;
        regex_search(entry, matcherName, reName);
        if (!matcherName.empty())
            name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
        else
            continue;
        labelsMap.insert(pair<int, string>(id, name));
    }
    return Status::OK();
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;
    
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);
    
    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);
    
    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    
    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));
    
    outTensor = outTensors.at(0);
    return Status::OK();
}

/** Draw bounding box and add caption to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled=true) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
	if (label == "Goods")
	{
		cv::rectangle(image, tl, br, cv::Scalar(0, 255, 0), 6);
	}
	else if (label == "Human")
	{
		cv::rectangle(image, tl, br, cv::Scalar(255, 255, 255), 6);
	}
	else if (label == "Sun")
	{
		cv::rectangle(image, tl, br, cv::Scalar(230, 220, 150), 6);
	}
	else 
	{
		cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 6);
	}
    
    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 10;
    string scoreString = to_string(scoreRounded).substr(0, 4) + "%";
    string caption = label + " (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 25;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
	if (label == "Goods")
	{
		cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 0), -1);
		cv::putText(image, caption, textCorner, FONT_ITALIC, 0.8, cv::Scalar(0, 0, 0),2, CV_AA);
	}
	else if (label == "Human")
	{
		cv::rectangle(image, tl, brRect, cv::Scalar(255, 255, 255), -1);
		cv::putText(image, caption, textCorner, FONT_ITALIC, 0.8, cv::Scalar(0, 0, 0), 2, CV_AA);
	}
	
	else if (label == "Sun")
	{
		cv::rectangle(image, tl, brRect, cv::Scalar(230, 220, 150), -1);
		cv::putText(image, caption, textCorner, FONT_ITALIC, 0.8, cv::Scalar(0, 0, 0),2,CV_AA);
	}
	else
	{
		cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
		cv::putText(image, caption, textCorner, FONT_ITALIC, 0.8, cv::Scalar(0, 0, 0),2, CV_AA);
	}
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              map<int, string> &labelsMap,
                              vector<size_t> &idxs) {
    for (int j = 0; j < idxs.size(); j++)
        drawBoundingBoxOnImage(image,
                               boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                               boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                               scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}

/** Calculate intersection-over-union (IOU) for two given bbox Rects.
 */
double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);
    
    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;
    return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
 *  and IOU <= thresholdIOU with others).
 */
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);

    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }

    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}
```
使用
```
AI_result = deeplearning_detector_.DL_Detector(image_);
```
所有结果都存在vector（AI_result）中



____
## 二、C#

### 1、计算模块

#### 1.1、用最小二乘法拟合二元多次曲线
> 引用：http://blog.sina.com.cn/s/blog_6e51df7f0100thie.html

```
///<summary>
        ///用最小二乘法拟合二元多次曲线
        ///例如y=ax+b
        ///其中MultiLine将返回a，b两个参数。
        ///a对应MultiLine[1]
        ///b对应MultiLine[0]
        ///</summary>
        ///<param name="arrX">已知点的x坐标集合</param>
        ///<param name="arrY">已知点的y坐标集合</param>
        ///<param name="length">已知点的个数</param>
        ///<param name="dimension">方程的最高次数</param>
        public double[] MultiLine(double[] arrX, double[] arrY, int length, int dimension)//二元多次线性方程拟合曲线
        {
            int n = dimension + 1;                  //dimension次方程需要求 dimension+1个 系数
            double[,] Guass = new double[n, n + 1];      //高斯矩阵 例如：y=a0+a1*x+a2*x*x
            for (int i = 0; i < n; i++)
            {
                int j;
                for (j = 0; j < n; j++)
                {
                    Guass[i, j] = SumArr(arrX, j + i, length);
                }
                Guass[i, j] = SumArr(arrX, i, arrY, 1, length);
            }

            return ComputGauss(Guass, n);
    
        }
        private double SumArr(double[] arr, int n, int length) //求数组的元素的n次方的和
        {
            double s = 0;
            for (int i = 0; i < length; i++)
            {
                if (arr[i] != 0 || n != 0)
                    s = s + Math.Pow(arr[i], n);
                else
                    s = s + 1;
            }
            return s;
        }
        private double SumArr(double[] arr1, int n1, double[] arr2, int n2, int length)
        {
            double s = 0;
            for (int i = 0; i < length; i++)
            {
                if ((arr1[i] != 0 || n1 != 0) && (arr2[i] != 0 || n2 != 0))
                    s = s + Math.Pow(arr1[i], n1) * Math.Pow(arr2[i], n2);
                else
                    s = s + 1;
            }
            return s;
        }
        private double[] ComputGauss(double[,] Guass, int n)
        {
            int i, j;
            int k, m;
            double temp;
            double max;
            double s;
            double[] x = new double[n];
    
            for (i = 0; i < n; i++) x[i] = 0.0;//初始化
    
            for (j = 0; j < n; j++)
            {
                max = 0;
    
                k = j;
                for (i = j; i < n; i++)
                {
                    if (Math.Abs(Guass[i, j]) > max)
                    {
                        max = Guass[i, j];
                        k = i;
                    }
                }
    
                if (k != j)
                {
                    for (m = j; m < n + 1; m++)
                    {
                        temp = Guass[j, m];
                        Guass[j, m] = Guass[k, m];
                        Guass[k, m] = temp;
                    }
                }
    
                if (0 == max)
                {
                    // "此线性方程为奇异线性方程" 
                    return x;
                }
    
                for (i = j + 1; i < n; i++)
                {
                    s = Guass[i, j];
                    for (m = j; m < n + 1; m++)
                    {
                        Guass[i, m] = Guass[i, m] - Guass[j, m] * s / (Guass[j, j]);
                    }
                }
            }//结束for (j=0;j<n;j++)
    
            for (i = n - 1; i >= 0; i--)
            {
                s = 0;
                for (j = i + 1; j < n; j++)
                {
                    s = s + Guass[i, j] * x[j];
                }
                x[i] = (Guass[i, n] - s) / Guass[i, i];
            }
            return x;
        }//返回值是函数的系数



//主程序调用
double[] arrX = new double[] { 21614, 17852, 14496, 10765, 7014, 3224 };
double[] arrY = new double[] { 6.0715, 5.005, 4.0495, 2.9945, 1.93, 0.849 };
double[] result = MultiLine(arrY, arrX, 6, 1);
      

```


##  三、Python

```

```

## 四、MarkDown
注意：此只支持Typora的markdown格式，对于cmd Markdown不能全部支持（会做说明）
> 引用：https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#cmd-markdown
> 引用：https://www.jianshu.com/p/191d1e21f7ed
> 引用：https://blog.csdn.net/SIMBA1949/article/details/79001226

***
### 1、标题

在想要设置为标题的文字前面加#来表示
 一个#是一级标题，二个#是二级标题，以此类推。支持六级标题。

***
### 2、段落与字体

#### 2.1 加粗
要加粗的文字左右分别用两个*号包起来
**这是加粗的文字**

#### 2.2 斜体
要倾斜的文字左右分别用一个* 号包起来
*这是倾斜的文字*`

#### 2.3 斜体加粗
要倾斜和加粗的文字左右分别用三个 * 号包起来
***这是斜体加粗的文字***

注：同样可以用____来表示，*和_在Markdown中一致

#### 2.4 删除线
要加删除线的文字左右分别用两个~~号包起来

~~这是加删除线的文字~~

#### 2.5 下划线

<u>下划线</u>

#### 2.6 设置字体颜色，文字大小等
```
<font color='color:文字颜色;font-size:文字大小</font>
```
<font color=red size=4 >我是red</font>

#### 2.7 注脚

使用 [^keyword] 表示注脚。

这是一个注脚[^footnote]的样例。

这是第二个注脚[^2]的样例。

[^footnote]: 这是一个 *注脚* 的 **文本**。

[^footnote]:这是另一个 *注脚* 的 **文本**


#### 2.8 分割线

******

___________

--------


#### 2.9 引用

引用的使用格式

>+空格






- [ ] **Markdown 开发**
    - [ ] 改进 Cmd 渲染算法，使用局部渲染技术提高渲染效率
    - [ ] 支持以 PDF 格式导出文稿
    - [ ] 新增Todo列表功能 
    - [ ] 改进 LaTex 功能
        - [x] 修复 LaTex 公式渲染问题
        - [ ] 新增 LaTex 公式编号功能 
- [ ] **七月旅行准备**
    - [ ] 准备邮轮上需要携带的物品
    - [ ] 浏览日本免税店的物品
    - [ ] 购买蓝宝石公主号七月一日的船票





```Mermaid
sequence
    Alice->John: Hello John, how are you?
    loop every minute
        John-->Alice: Great!
    end
```

$$
\sum_{i=1}^n a_i=0
$$

$$
f(x_1,x_x,\ldots,x_n) = x_1^2 + x_2^2 + \cdots + x_n^2
$$

$$
\sum^{j-1}_{k=0}{\widehat{\gamma}_{kj} z_k}
$$






```

```

```

```