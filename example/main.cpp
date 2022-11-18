#include "ShapeMatch.h"

void DrawContours(IplImage* source, CvPoint Result, CvPoint* Contours, int ContoursSize, CvScalar color, int lineWidth)
{
	CvPoint point;
	int x = Result.x;
	int y = Result.y;
	cvLine(source, cvPoint(x, y - 5), cvPoint(x, y + 5), CV_RGB(0, 0, 255), lineWidth);
	cvLine(source, cvPoint(x - 5, y), cvPoint(x + 5, y), CV_RGB(0, 0, 255), lineWidth);
	for (int i = 0; i < ContoursSize; i++)
	{
		point.x=Contours[i].x + x;
		point.y=Contours[i].y + y;
		cvLine(source,point,point,color,lineWidth);
	}
}

void DrawContours(IplImage* source, CvPoint* Contours, int ContoursSize, CvScalar color, int lineWidth)
{
	CvPoint point;
	int x = source->width / 2;
	int y = source->height / 2;
	cvLine(source, cvPoint(x , y-5), cvPoint(x, y+5),CV_RGB( 0, 0, 255 ),lineWidth);
	cvLine(source, cvPoint(x-5 , y), cvPoint(x+5, y),CV_RGB( 0, 0, 255 ),lineWidth);
	for (int i = 0; i < ContoursSize; i++)
	{
		point.x=Contours[i].x ;
		point.y=Contours[i].y ;
		cvLine(source,point,point,color,lineWidth);
	}
}


int main(int argc, char** argv[])
{
	CShapeMatch SM;
	IplImage* templateImage = cvLoadImage("E://Xiongxinxin//ShapeMatch//data//1.bmp", -1);
	IplImage* searchImage = cvLoadImage("E://Xiongxinxin//ShapeMatch//data//g.bmp", -1);
	if (!searchImage || !templateImage)
	{
		cout << " 图片加载失败！\n";
		system("pause");
		return 0;
	}
	
	CvSize templateSize = cvSize(templateImage->width, templateImage->height);
	CvSize searchSize = cvSize(searchImage->width, searchImage->height);
	IplImage* grayTemplateImg = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);
	IplImage* graySearchImg = cvCreateImage(searchSize, IPL_DEPTH_8U, 1);

	// 求解金子塔阶数
	int dim = min(templateImage->width, templateImage->height);  
	int numoctaves = (int) (log((double) dim) / log(2.0)) - 2;		  
	numoctaves = min(numoctaves, 7);						

	// 将图像转换为单通道
	if(templateImage->nChannels == 3) cvCvtColor(templateImage, grayTemplateImg, CV_RGB2GRAY);
	else cvCopy(templateImage, grayTemplateImg);
	if (searchImage->nChannels == 3) cvCvtColor(searchImage, graySearchImg, CV_RGB2GRAY);
	else cvCopy(searchImage, graySearchImg);

	// 设置模板模型参数
	shape_model ModelID;
	ModelID.m_AngleStart	= -180;									//起始角度
	ModelID.m_AngleStop  	= 180;									//终止角度
	ModelID.m_AngleStep		= 1;									//角度步长
	ModelID.m_Contrast		= 80;									//高阈值
	ModelID.m_MinContrast	= 10;									//低阈值
	ModelID.m_NumLevels		= 2;									//金字塔级数
	ModelID.m_Granularity   = 1;									//颗粒度
	ModelID.m_ImageWidth    = grayTemplateImg->width;
	ModelID.m_ImageHeight   = grayTemplateImg->height;

	// 提取模板图像有效边缘点并绘制
	edge_list EdgeList;
	EdgeList.EdgePiont = (CvPoint*)malloc(grayTemplateImg->width * grayTemplateImg->height * sizeof(CvPoint));
	SM.train_shape_model(grayTemplateImg, ModelID.m_Contrast, ModelID.m_MinContrast, ModelID.m_Granularity, &EdgeList);
	DrawContours(templateImage, EdgeList.EdgePiont, EdgeList.ListSize, CV_RGB(255, 0, 0), 1);
	cvNamedWindow("Template", CV_WINDOW_AUTOSIZE);
	cvShowImage("Template", templateImage);
	cvWaitKey(0);

	SM.initial_shape_model(&ModelID, grayTemplateImg->width, grayTemplateImg->height, EdgeList.ListSize);
	free(EdgeList.EdgePiont);

	cout<< "\n Search Model Program\n"; 
	cout<< " ------------------------------------\n";
	cout<< " 角度范围：" <<ModelID.m_AngleStart <<"°~ "<<ModelID.m_AngleStop<<"°\n";

	/* Create shape model file*/
	clock_t start_time = clock();
 	bool IsInial = SM.create_shape_model(grayTemplateImg, &ModelID);
	clock_t finish_time = clock();

	double total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;
	cout<< " ------------------------------------\n";
	cout<<" Create Time = "<<total_time*1000<<"ms\n";

	// 设置模板匹配参数
	int		NumMatch	= 4;			//匹配个数
	float	MinScore	= 0.7f;			//最小评分
	float	Greediness	= 0.7f;			//贪婪度

	MatchResultA Result[10];
	memset(Result, 0, 10 * sizeof(MatchResultA));
	cout<< " ------------------------------------\n";
	if(IsInial)
	{
		start_time = clock();
		SM.find_shape_model(graySearchImg, &ModelID, MinScore, NumMatch, Greediness, Result);
		finish_time = clock();
	}
	else
		printf(" Create model failed!\n");

	total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;
	cout<<" Find Time = "<<total_time*1000<<"ms\n";
	cout<< " ------------------------------------\n";

	/* Draw contours in search image. */
	ShapeInfo *temp = ModelID.m_pShapeInfoTmpVec;
	CvPoint * Contours = temp->Coordinates;
	int count = temp->NoOfCordinates;
	temp = ModelID.m_pShapeInfoTmpVec;
	Contours = temp->Coordinates;
	count = 0;
	for (int n = 0; n <NumMatch; n++)
	{
		if (Result[n].ResultScore != 0)
		{
			for (int i = 0; i < temp[0].AngleNum; i++)
			{
				if (temp[i].Angel == Result[n].Angel)
				{
					Contours = temp[i].Coordinates;
					count = temp[i].NoOfCordinates;
					break;
				}
			}
			printf(" Location:(%d, %d) Angle: %d Score: %.4f\n", Result[n].CenterLocX, Result[n].CenterLocY, Result[n].Angel, Result[n].ResultScore);
			DrawContours(searchImage, cvPoint(Result[n].CenterLocX, Result[n].CenterLocY), Contours, count, CV_RGB( 0, 255, 0 ),1);
		}
	}
	SM.release_shape_model(&ModelID);
	
	//Display result
	cvNamedWindow("Search Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Search Image", searchImage);
	cvWaitKey(0);
	//Wait for both windows to be closed before releasing images
	
	cvDestroyWindow("Search Image");
	cvDestroyWindow("Template");
	cvReleaseImage(&graySearchImg);
	cvReleaseImage(&searchImage);
	cvReleaseImage(&grayTemplateImg);	
	cvReleaseImage(&templateImage);

	return 0;
}