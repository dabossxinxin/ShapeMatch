#include "ShapeMatchV2.h"

void DrawContours(cv::Mat& source, cv::Point& Result, std::vector<cv::Point>& Contours, int ContoursSize, cv::Scalar& color, int lineWidth)
{
	cv::Point point;
	int x = Result.x;
	int y = Result.y;
	cv::line(source, cv::Point(x, y - 5), cv::Point(x, y + 5), CV_RGB(0, 0, 255), lineWidth);
	cv::line(source, cv::Point(x - 5, y), cv::Point(x + 5, y), CV_RGB(0, 0, 255), lineWidth);
	for (int i = 0; i < ContoursSize; i++)
	{
		point.x=Contours[i].x + x;
		point.y=Contours[i].y + y;
		cv::line(source,point,point,color,lineWidth);
	}
}

void DrawContours(cv::Mat& source, std::vector<cv::Point> Contours, int ContoursSize, cv::Scalar color, int lineWidth)
{
	cv::Point point;
	int x = source.cols / 2;
	int y = source.rows / 2;
	cv::line(source, cv::Point(x, y - 5), cv::Point(x, y + 5), cv::Scalar(0, 0, 255), lineWidth);
	cv::line(source, cv::Point(x - 5, y), cv::Point(x + 5, y), cv::Scalar(0, 0, 255), lineWidth);
	for (int i = 0; i < ContoursSize; i++)
	{
		point.x = Contours[i].x;
		point.y = Contours[i].y;
		cv::line(source, point, point, color, lineWidth);
	}
}


int main(int argc, char** argv[])
{
	CShapeMatchV2 SM;
	cv::Mat templateImage = cv::imread("D://Code//ShapeMatch//data//qfnTemplateImage.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat searchImage = cv::imread("D://Code//ShapeMatch//data//qfnSearchImage.bmp", cv::IMREAD_GRAYSCALE);
	if (!searchImage.data || !templateImage.data)
	{
		std::cout << " 图片加载失败！\n";
		system("pause");
		return 0;
	}

	cv::Mat templateImageRGB = cv::Mat(templateImage.rows, templateImage.cols, CV_8UC3);
	cv::Mat searchImageRGB = cv::Mat(templateImage.rows, templateImage.cols, CV_8UC3);
	cv::cvtColor(templateImage, templateImageRGB, cv::COLOR_GRAY2RGB);
	cv::cvtColor(searchImage, searchImageRGB, cv::COLOR_GRAY2RGB);

	cv::Size templateSize = cv::Size(templateImage.cols, templateImage.rows);
	cv::Size searchSize = cv::Size(searchImage.cols, searchImage.rows);

	// 求解金子塔阶数
	int dim = std::min(templateImage.cols, templateImage.rows);
	int numoctaves = (int)(log((double)dim) / log(2.0)) - 2;
	numoctaves = std::min(numoctaves, 7);

	// 设置模板模型参数
	ShapeModelV2 ModelID;
	ModelID.m_AngleStart = -10;									//起始角度
	ModelID.m_AngleStop = 10;									//终止角度
	ModelID.m_AngleStep = 1;									//角度步长
	ModelID.m_Contrast = 160;									//高阈值
	ModelID.m_MinContrast = 100;								//低阈值
	ModelID.m_NumLevels = 7;									//金字塔级数
	ModelID.m_Granularity = 1;									//颗粒度
	ModelID.m_ImageWidth = templateImage.cols;
	ModelID.m_ImageHeight = templateImage.rows;

	// 提取模板图像有效边缘点并绘制
	EdgeListV2 EdgeList;
	SM.train_shape_model(templateImage, ModelID.m_Contrast, ModelID.m_MinContrast, ModelID.m_Granularity, EdgeList);

	DrawContours(templateImageRGB, EdgeList.EdgePoint, EdgeList.ListSize, CV_RGB(0, 0, 255), 2);
	cv::namedWindow("Template", cv::WINDOW_AUTOSIZE);
	cv::imshow("Template", templateImageRGB);

	SM.initial_shape_model(ModelID, templateImage.cols, templateImage.rows, EdgeList.ListSize);

	std::cout << "\n Search Model Program\n";
	std::cout << " ------------------------------------\n";
	std::cout << " 角度范围：" << ModelID.m_AngleStart << "°~ " << ModelID.m_AngleStop << "°\n";

	// 创建形状模板文件
	clock_t start_time = clock();
	bool IsInial = SM.create_shape_model(templateImage, ModelID);
	clock_t finish_time = clock();

	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << " ------------------------------------\n";
	std::cout << " Create Time = " << total_time * 1000 << "ms\n";

	// 设置模板匹配参数
	int		NumMatch = 1;			//匹配个数
	float	MinScore = 0.6f;		//最小评分
	float	Greediness = 0.6f;		//贪婪度

	MatchResultV2* Result = (MatchResultV2*)malloc(NumMatch * sizeof(MatchResultV2));
	memset(Result, 0, NumMatch * sizeof(MatchResultV2));

	for (int it = 0; it < 1; ++it) {
		std::cout << " ------------------------------------\n";
		if (IsInial)
		{
			start_time = clock();
			SM.find_shape_model(searchImage, ModelID, MinScore, NumMatch, Greediness, Result);
			finish_time = clock();
		}
		else
			printf(" Create model failed!\n");

		total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << " Find Time = " << total_time * 1000 << "ms\n";
		std::cout << " ------------------------------------\n";
	}

	// 在搜索图像中绘制特征
	ShapeInfoV2* temp = nullptr;
	temp = ModelID.m_pShapeInfoPyr[0];
	auto Contours = temp->Coordinates;
	int count = 0;
	for (int n = 0; n < NumMatch; n++) {
		if (Result[n].ResultScore != 0) {
			for (int i = 0; i < temp[0].AngleNum; i++) {
				if (temp[i].Angle == Result[n].Angle) {
					Contours = temp[i].Coordinates;
					count = temp[i].NoOfCordinates;
					break;
				}
			}
			printf(" Location:(%d, %d) Angle: %d Score: %.4f\n", Result[n].CenterLocX, Result[n].CenterLocY, Result[n].Angle, Result[n].ResultScore);
			DrawContours(searchImageRGB, cv::Point(Result[n].CenterLocX, Result[n].CenterLocY), Contours, count, CV_RGB(0, 255, 0), 2);
		}
	}
	SM.release_shape_model(ModelID);
	if (Result) std::free(Result);
	
	//Display result
	cv::namedWindow("Search Image", cv::WINDOW_FREERATIO);
	cv::imshow("Search Image", searchImageRGB);
	cv::waitKey(0);

	cv::destroyWindow("Search Image");
	cv::destroyWindow("Template");

	return 0;
}