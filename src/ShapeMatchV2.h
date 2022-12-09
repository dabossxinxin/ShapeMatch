#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#define SAVE_IMG
#define	MAXTARGETNUM 64
#define SCORE_RATIO 0.9

const int K_CosineTable[24] =
{
	8192, 8172, 8112, 8012, 7874, 7697,
	7483, 7233, 6947, 6627, 6275, 5892, 
	5481, 5043, 4580, 4096, 3591, 3068,
	2531, 1981, 1422, 856,   285, -285
};

struct MatchResultV2
{
	int 				Angle;						//匹配角度
	int 				CenterLocX;					//匹配参考点X坐标
	int					CenterLocY;					//匹配参考点Y坐标
	float 				ResultScore;				//匹配得分
	
	MatchResultV2() : Angle(0), CenterLocX(0),
		CenterLocY(0), ResultScore(0.0)
	{

	}
};

struct ShapeInfoV2
{
	cv::Point					ReferPoint;					//模板重心坐标
	std::vector<cv::Point>		Coordinates;				//模板坐标数组
	std::vector<cv::Point2f>	EdgeDirection;				//梯度方向
	int 						ImgWidth;					//图像宽度
	int							ImgHeight;					//图像高度
	int							NoOfCordinates;				//轮廓点个数
	int							Angle;						//旋转角度
	int							PyLevel;					//金字塔级别
	int							AngleNum;					//角度个数

	ShapeInfoV2() : AngleNum(0),PyLevel(0),Angle(0),NoOfCordinates(0),
		ImgHeight(0),ImgWidth(0),ReferPoint(cv::Point(0,0))
	{
		Coordinates.clear();
		EdgeDirection.clear();
	}
};

struct ShapeModelV2
{
	int							ID;							//模板ID
	int 						m_NumLevels;				//金字塔级数
	int 						m_Contrast;					//高阈值
	int 						m_MinContrast;				//低阈值
	int 						m_Granularity;				//边缘颗粒度
	int 						m_AngleStart;				//模板旋转起始角度
	int 						m_AngleStop;				//模板旋转终止幅度
	int 						m_AngleStep;				//角度步长
	int							m_ImageWidth;				//原模板图像宽度
	int							m_ImageHeight;				//原模板图像高度
	bool						m_IsInited;					//初始化标志
	std::map<int, ShapeInfoV2*>	m_pShapeInfoPyr;			//图像形状信息

	ShapeModelV2() : ID(0), m_NumLevels(0), m_Contrast(0), m_MinContrast(0),
		m_Granularity(0), m_AngleStart(0), m_AngleStep(0), m_AngleStop(0),
		m_ImageWidth(0), m_ImageHeight(0), m_IsInited(false)
	{
		m_pShapeInfoPyr.clear();
	}
};

struct SearchRegionV2
{
	int 				StartX;						//X方向起点
	int 				StartY;						//y方向起点
	int 				EndX;						//x方向终点
	int 				EndY;						//y方向终点
	int 				AngleRange;					//搜索角度数目
	int					AngleStart;					//搜索预先角度
	int					AngleStop;					//搜索终止角度
	int					AngleStep;					//搜索角度步长
};

// 边界点列表
struct EdgeListV2
{
	std::vector<cv::Point>			EdgePoint;					//边缘坐标数组
	int 							ListSize;					//数组大小

	EdgeListV2() : ListSize(0)
	{
		EdgePoint.clear();
	}
};

class ShapeMatchTool
{
public:
	int  ShiftCos(int y);

	int  ShiftSin(int y);

	float Q_rsqrt(float number);

	float new_rsqrt(float f);

	void QuickSort(MatchResultV2* s, int l, int r);

	int ConvertLength(int LengthSrc);

	int factorial(int num, int level);
};

class CShapeMatchV2 :public ShapeMatchTool
{
public:
	CShapeMatchV2();
	~CShapeMatchV2();

	void gaussian_filter(
		unsigned char* corrupted, 
		unsigned char* smooth,
		const int width, 
		const int height);

	void gen_rectangle(
		const cv::Mat& Image, 
		cv::Mat& ModelRegion, 
		const int Row,
		const int Column);

	void board_image(
		const cv::Mat& SrcImg, 
		cv::Mat& ImgBordered, 
		const int xOffset, 
		const int yOffset);

	void rotate_image(
		const cv::Mat& srcImage,
		cv::Mat& dstImage,
		const int Angle);

	void image_pyramid(
		unsigned char* SrcImgData,
		const int srcWidth,
		const int srcHeight,
		const int PyrNum,
		unsigned char* OutImgData);

	void initial_shape_model(
		ShapeModelV2& ModelID,
		const int Width, 
		const int Height, 
		const int EdgeSize);
	
	bool release_shape_model(
		ShapeModelV2& ModelID);
	
	void extract_shape_info(
		unsigned char* ImageData, 
		ShapeInfoV2* ShapeInfoData, 
		const int Contrast, 
		const int MinContrast, 
		const int PointReduction, 
		unsigned char* MaskImgData);

	bool build_model_list(
		ShapeInfoV2* ShapeInfoVec, 
		unsigned char* ImageData, 
		unsigned char* MaskData, 
		const int Width, 
		const int Height,
		const int Contrast, 
		const int MinContrast,
		const int Granularity);

	void train_shape_model(
		const cv::Mat& Image, 
		const int Contrast, 
		const int MinContrast,
		const int PointReduction, 
		EdgeListV2& EdgeList);

	bool create_shape_model(
		const cv::Mat& Template,
		ShapeModelV2& ModelID);

	void shape_match(
		unsigned char* SearchImage, 
		ShapeInfoV2 *ShapeInfoVec, 
		const int Width, 
		const int Height,
		int* NumMatches,
		const int Contrast,
		const int MinContrast, 
		const float MinScore, 
		const float Greediness,
		SearchRegionV2* SearchRegion,
		MatchResultV2* ResultList);

	void shape_match_accurate(
		unsigned char* SearchImage,
		ShapeInfoV2* ShapeInfoVec,
		const int Width,
		const int Height,
		const int Contrast,
		const int MinContrast,
		const float MinScore,
		const float Greediness,
		SearchRegionV2* SearchRegion,
		MatchResultV2* ResultList);

	void find_shape_model(
		const cv::Mat& Image, 
		ShapeModelV2& ModelID, 
		const float MinScore,
		const int NumMatches, 
		const float Greediness,
		MatchResultV2* ResultList);
private:
	void initial_shape_model_impl(
		ShapeModelV2& ModelID,
		const int Length,
		const int EdgeSize,
		const int PyrLevel
	);

	void find_shape_model_impl(
		ShapeModelV2& ModelID,
		const int PyrLevel,
		const int NumMatches,
		const int BorderedWidth,
		const int BorderedHeight,
		unsigned char* BorderedImgData,
		const int xOffset,
		const int yOffset,
		const float MinScore,
		const float Greediness,
		unsigned char* pImagePyrAll,
		MatchResultV2* ResultList
	);

	bool create_shape_model_impl(
		ShapeModelV2& ModelID,
		const int WidthPyr,
		const int HeightPyr,
		unsigned char* pImageDataPyr,
		unsigned char* pMaskDataPyr,
		const int PyrLevel
	);

	void image_pyramid_impl(
		unsigned char* SrcImgData,
		const int srcWidth,
		const int srcHeight,
		const int PyrLevel,
		const int PyrNum,
		unsigned char* OutImgData
	);

	void release_shape_model_impl(
		ShapeModelV2& ModelID,
		const int PyrLevel
	);

	void download_pyr_img(
		const std::string& filename,
		const int imgWidth,
		const int imgHeight,
		unsigned char* imgData
	);
};


template <class T>
class MinHeap
{
public:
	MinHeap() {
		m_iQueueSize = 0;
		m_vPQueue.resize(2);
	}

	~MinHeap() {
		m_iQueueSize = 0;
		m_vPQueue.clear();
	}

	void push(const T& element) {
		if (m_iQueueSize == 0) {
			m_vPQueue.resize(2);
		}
		if (m_iQueueSize == m_vPQueue.size() - 1) {
			ChangeLength();
		}

		int currentNode = ++m_iQueueSize;
		while (currentNode != 1 && element.ResultScore < m_vPQueue[currentNode / 2].ResultScore) {
			m_vPQueue[currentNode] = m_vPQueue[currentNode / 2];
			currentNode /= 2;
		}
		m_vPQueue[currentNode] = element;
	}

	void pop() {
		int deleteIndex = m_iQueueSize;
		T lastElement = m_vPQueue[m_iQueueSize--];

		int currentNode = 1, chirld = 2;
		while (chirld <= m_iQueueSize) {
			if (chirld < m_iQueueSize && m_vPQueue[chirld].ResultScore > m_vPQueue[chirld + 1].ResultScore) {
				chirld++;
			}
			if (lastElement.ResultScore <= m_vPQueue[chirld].ResultScore) {
				break;
			}

			m_vPQueue[currentNode] = m_vPQueue[chirld];
			currentNode = chirld;
			chirld *= 2;
		}
		m_vPQueue[currentNode] = lastElement;
		//m_vPQueue.erase(m_vPQueue.begin() + deleteIndex);
		m_vPQueue[deleteIndex] = T();
	}

	int size() const {
		return m_iQueueSize;
	}

	T top() {
		if (!empty()) 
			return m_vPQueue[1];
		else {
			std::cerr << "priority queue size error." << std::endl;
			return T();
		}
	}

	bool empty() {
		return m_iQueueSize == 0;
	}

	std::vector<T> GetElement() const{
		return m_vPQueue;
	}

private:
	std::vector<T>	m_vPQueue;
	int				m_iQueueSize;
	
	inline void ChangeLength() {
		auto tmpQueue = m_vPQueue;
		m_vPQueue.resize(2 * tmpQueue.size());
		for (int it = 0; it < tmpQueue.size(); ++it)
			m_vPQueue[it] = tmpQueue[it];
	}
};

template <class T>
class PriorityQueue {
public:
	PriorityQueue() {
		m_hPQueue = MinHeap<T>();
		m_iQueueSize = 0;
	}

	void clear() {
		m_hPQueue.~MinHeap();
		m_iQueueSize = 0;
	}

	void resize(const int size) {
		m_iQueueSize = size;
	}

	void push(const T& element) {
		if (m_iQueueSize == 0)
			return;

		if (m_hPQueue.size() < m_iQueueSize) {
			m_hPQueue.push(element);
		}
		else {
			if (element.ResultScore > m_hPQueue.top().ResultScore) {
				m_hPQueue.pop();
				m_hPQueue.push(element);
			}
		}
	}

	std::vector<T> GetElement() const{
		return m_hPQueue.GetElement();
	}

	int size() const{
		return m_hPQueue.size();
	}

private:
	MinHeap<T>	m_hPQueue;
	int			m_iQueueSize;
};