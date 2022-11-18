#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

#define		MAX_NUM_INSTANCES		300				//最大目标个数
#define		MIN_NUM_LEVELS			0				//最小金字塔级数
#define		MAX_NUM_LEVELS			5				//最大金字塔级数

const int K_CosineTable[24] =
{
	8192, 8172, 8112, 8012, 7874, 7697,
	7483, 7233, 6947, 6627, 6275, 5892, 
	5481, 5043, 4580, 4096, 3591, 3068,
	2531, 1981, 1422, 856,   285, -285
};

//匹配结果结构体
struct MatchResultA
{
	int 				Angel;						//匹配角度
	int 				CenterLocX;					//匹配参考点X坐标
	int					CenterLocY;					//匹配参考点Y坐标
	float 				ResultScore;				//匹配得分
};

//特征信息结构体
struct ShapeInfo
{
	CvPoint				ReferPoint;					//模板重心坐标
	CvPoint				*Coordinates;				//模板坐标数组
	float				*EdgeMagnitude;				//梯度导数
	short				*EdgeDerivativeX;			//X方向梯度
	short				*EdgeDerivativeY;			//Y方向梯度
	int 				ImgWidth;					//图像宽度
	int					ImgHeight;					//图像高度
	int					NoOfCordinates;				//轮廓点个数
	int					Angel;						//旋转角度
	int					PyLevel;					//金字塔级别
	int					AngleNum;					//角度个数
};

//模板文件结构体
struct shape_model
{
	int					ID;							//模板ID
	int 				m_NumLevels;				//金字塔级数
	int 				m_Contrast;					//高阈值
	int 				m_MinContrast;				//低阈值
	int 				m_Granularity;				//边缘颗粒度
	int 				m_AngleStart;				//模板旋转起始角度
	int 				m_AngleStop;				//模板旋转终止幅度
	int 				m_AngleStep;				//角度步长
	int					m_ImageWidth;				//原模板图像宽度
	int					m_ImageHeight;				//原模板图像高度
	bool				m_IsInited;					//初始化标志

	// TODO 将不同金字塔指针调整在map中
	ShapeInfo			*m_pShapeInfoPyd1Vec;		//模板金字塔第1级图像的边缘信息
	ShapeInfo			*m_pShapeInfoPyd2Vec;		//模板金字塔第2级图像的边缘信息
	ShapeInfo			*m_pShapeInfoPyd3Vec;		//模板金字塔第3级图像的边缘信息
	ShapeInfo			*m_pShapeInfoTmpVec;		//原模板图像的边缘信息
};

//搜索区域
struct search_region
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

//边界点列表
struct edge_list
{
	CvPoint				*EdgePiont;					//边缘坐标数组
	int 				ListSize;					//数组大小

};

class CShapeMatch
{
public:
	CShapeMatch(void);
	~CShapeMatch(void);

	// 对图像进行高斯滤波
	void gaussian_filter(uint8_t* corrupted, uint8_t* smooth, int width, int height);
		/*--------------------------------------------------------------------------------------------*/
	/*	函数名：gen_rectangle
		函数功能：生成ROI
		输入变量：Image 输入图像, Row1 左上角点的横坐标, Column1 左上角点的纵坐标
		返回变量：
		注释：		*/

	void gen_rectangle(IplImage *Image, IplImage *ModelRegion, int Row1, int Column1);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：gen_rectangle
		函数功能：生成ROI
		输入变量：Image 输入图像, Row1 左上角点的横坐标, Column1 左上角点的纵坐标
		返回变量：
		注释：		*/

	void board_image(IplImage *SrcImg, IplImage *ImgBordered, int32_t xOffset, int32_t yOffset);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：board_image
		函数功能：图像边界扩展
		输入变量：SrcImage 输入图像, DstImage 输出图像
		返回变量：
		注释：		*/

	void rotate_image (uint8_t *SrcImgData, uint8_t *MaskImgData, int srcWidth, int srcHeight, uint8_t *DstImgData, uint8_t *MaskRotData, int dstWidth, int dstHeight, int Angle);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：rotate_image
		函数功能：图像旋转函数
		输入变量：SrcImage 输入图像, DstImage 输出图像, Angle旋转角度
		返回变量：DstImgData 旋转后的图像
		注释：		*/

	// 对图像进行指定角度旋转
	void rotateImage (IplImage* srcImage, IplImage* dstImage, int Angle);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：rotate_image
		函数功能：图像旋转函数
		输入变量：SrcImage 输入图像, DstImage 输出图像, Angle旋转角度
		返回变量：DstImgData 旋转后的图像
		注释：		*/

	// 以2*2形式构建图像金字塔
	void image_pyramid(uint8_t *SrcImgData,  int srcWidth, int srcHeight, uint8_t *OutImgData);

	// 以双线性插值构建图像金字塔
	void imagePyramid(uint8_t *SrcImgData,  int srcWidth, int srcHeight, uint8_t *OutImgData);

	// 初始化模板资源
	void initial_shape_model(shape_model *ModelID, int Width, int Height, int EdgeSize);
	
	// 释放模板资源
	bool release_shape_model(shape_model *ModelID);
	
	// 提取图像形状信息
	void extract_shape_info(uint8_t *ImageData, ShapeInfo *ShapeInfoData, int Contrast, int MinContrast, int PointReduction, uint8_t *MaskImgData);

	// 创建角度模板序列
	bool build_model_list(ShapeInfo *ShapeInfoVec, uint8_t *ImageData, uint8_t *MaskData, int Width, int Height, int Contrast, int MinContrast, int Granularity);

	// 定位模板图像边缘点作为模板形状
	void train_shape_model(IplImage *Image, int Contrast, int MinContrast, int PointReduction, edge_list *EdgeList);

	// 创建匹配模板
	bool create_shape_model(IplImage *Template, shape_model *ModelID);

	void shape_match(uint8_t *SearchImage, ShapeInfo *ShapeInfoVec, int Width, int Height, int *NumMatches, int Contrast, int MinContrast, float MinScore, float Greediness, search_region *SearchRegion, MatchResultA *ResultList);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：shape_match
		函数功能：形状匹配函数
		输入变量：SearchImage 待搜索图像数据,  ShapeInfo 模板形状信息, Width 图像宽度, Height 图像高度, NumMatches 匹配目标数 MinScore 最小评分, Greediness 贪婪度, SearchRegion 搜索范围
		返回变量：ResultList 匹配结果
		注释：		*/
	
	void shape_match_accurate(uint8_t *SearchImage, ShapeInfo *ShapeInfoVec, int Width, int Height, int Contrast, int MinContrast, float MinScore, float Greediness, search_region *SearchRegion, MatchResultA *ResultList);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：shape_match_accurate
		函数功能：精确形状匹配函数
		输入变量：SearchImage 待搜索图像数据,  ShapeInfo 模板形状信息, Width 图像宽度, Height 图像高度, NumMatches 匹配目标数 MinScore 最小评分, Greediness 贪婪度, SearchRegion 搜索范围
		返回变量：ResultList 匹配结果
		注释：		*/

	void find_shape_model(IplImage *Image, shape_model *ModelID, float MinScore, int NumMatches, float Greediness, MatchResultA *ResultList);
	/*--------------------------------------------------------------------------------------------*/
	/*	函数名：find_shape_model
		函数功能：在搜索图中寻找模板目标
		输入变量：Image 待搜索图像,  ModelID 模板文件, AngleStart 起始角度, AngleExtent 角度范围, MinScore 最小评分, NumMatches 匹配目标数
		返回变量：Row 匹配参考点点X坐标, Column 匹配参考点点Y坐标, Angle 目标旋转角度, Score 评分
		注释：		*/

	// 余弦三角函数
	int  ShiftCos(int y);

	// 正弦三角函数
	int  ShiftSin(int y);
	
	// 求解平方根倒数
	float Q_rsqrt( float number );
	
	// 求解平方根倒数
	float new_rsqrt(float f);
	
	// 快速数组排序
	void QuickSort(MatchResultA *s, int l, int r);

	// 长度转换
	int ConvertLength(int LengthSrc);
};

