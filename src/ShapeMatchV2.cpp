#include "ShapeMatchV2.h"

#define	MAXTARGETNUM 64
#define MAXPIXELERROR 16

PriorityQueue<MatchResultV2> resultsPerDeg;
PriorityQueue<MatchResultV2> totalResultsTemp;

CShapeMatchV2::CShapeMatchV2()
{
}

CShapeMatchV2::~CShapeMatchV2()
{
}

void CShapeMatchV2::gaussian_filter(
	unsigned char* corrupted, 
	unsigned char* smooth, 
	const int width, 
	const int height)
{
	int templates[25] = 
	{ 1,  4,  7,  4, 1,   
	  4, 16, 26, 16, 4,   
	  7, 26, 41, 26, 7,  
	  4, 16, 26, 16, 4,   
	  1,  4,  7,  4, 1 };        

	memcpy(smooth, corrupted, width*height * sizeof(unsigned char));
	for (int j = 2; j < height - 2; ++j) {
		for (int i = 2; i < width - 2; ++i) {
			int sum = 0;  
			int index = 0;  
			for (int m = j - 2; m < j + 3; ++m) {
				for (int n = i - 2; n < i + 3; ++n) {
					sum += corrupted[m*width + n] * templates[index++];
				}  
			}  
			sum /= 273;
			if (sum > 255)  
				sum = 255;  
			smooth[j*width + i] = (unsigned char)sum;
		}  
	}  
}

void CShapeMatchV2::gen_rectangle(
	const cv::Mat& Image, 
	cv::Mat& ModelRegion, 
	const int Row, 
	const int Column)
{
	ModelRegion = cv::Mat(Image(cv::Rect(Row, Column, ModelRegion.cols, ModelRegion.rows))).clone();
	return;
}

void CShapeMatchV2::board_image(
	const cv::Mat& SrcImg,
	cv::Mat& ImgBordered, 
	const int xOffset,
	const int yOffset)
{
	const int ImgBorderedWStep = ImgBordered.cols;
	const int ImgFilteredWStep = SrcImg.cols;

	memset(ImgBordered.data, 0, ImgBordered.rows*ImgBordered.cols * sizeof(unsigned char));

	// 拷贝图像区域像素值
	for(int row = yOffset; row < SrcImg.rows + yOffset; ++row)								
		memcpy(ImgBordered.data + row * ImgBorderedWStep + xOffset, SrcImg.data + (row - yOffset)*ImgFilteredWStep, SrcImg.cols);

	// 填充图像上边界
	for(int row = yOffset; row >=0; --row)														
		memcpy(ImgBordered.data + row * ImgBorderedWStep + xOffset, SrcImg.data, SrcImg.cols);

	// 填充图像下边界
	for (int row = yOffset + SrcImg.rows; row < ImgBordered.rows; ++row)
		memcpy(ImgBordered.data + row * ImgBorderedWStep + xOffset, SrcImg.data + (SrcImg.rows - 1)*ImgFilteredWStep, SrcImg.cols);

	// 填充图像左边界
	for(int col = 0; col < xOffset; ++col) {
		for(int row = 0; row < ImgBordered.rows; ++row) {
			*(ImgBordered.data + row * ImgBorderedWStep + col) = *(ImgBordered.data + row * ImgBorderedWStep + xOffset);
		}
	}

	// 填充图像右边界
	for(int col = xOffset + SrcImg.cols; col < ImgBordered.cols; ++col) {
		for(int row = 0; row < ImgBordered.rows; ++row) {
			*(ImgBordered.data + row * ImgBorderedWStep + col) = *(ImgBordered.data + row * ImgBorderedWStep + xOffset + SrcImg.cols - 1);
		}
	}
}

void CShapeMatchV2::rotate_image(
	const cv::Mat& srcImage,
	cv::Mat& dstImage,
	const int Angle)
{
	cv::Size src_size = srcImage.size();
	cv::Size dst_size(dstImage.cols, dstImage.rows);

	cv::Point2f center = cv::Point2f(srcImage.cols*0.5f, srcImage.rows*0.5f);
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, double(Angle), 1.0);

	rot_mat.at<double>(0, 2) += dst_size.width*0.5f - center.x;
	rot_mat.at<double>(1, 2) += dst_size.height*0.5f - center.y;

	cv::warpAffine(srcImage, dstImage, rot_mat, dst_size);
}

void CShapeMatchV2::image_pyramid_impl(
	unsigned char* SrcImgData,
	const int srcWidth,
	const int srcHeight,
	const int PyrLevel,
	const int PyrNum,
	unsigned char* OutImgData
)
{
	int w = srcWidth >> PyrLevel;
	int h = srcHeight >> PyrLevel;
	int in_size = srcWidth * srcHeight;
	int factor = std::pow(2, PyrLevel);

	int offset = 0;
	for (int pyr = 2; pyr <= PyrLevel; ++pyr) {
		offset += in_size / std::pow(4, pyr - 1);
	}

	for (int row = 0; row < h; ++row) {
		for (int col = 0; col < w; ++col) {
			*(OutImgData + offset + row * w + col) =
				*(SrcImgData + factor * row*srcWidth + factor * col);
		}
	}
}

void CShapeMatchV2::image_pyramid(
	unsigned char* SrcImgData, 
	const int srcWidth, 
	const int srcHeight,
	const int PyrNum,
	unsigned char* OutImgData)
{
	if (PyrNum == 0) {
		return;
	}

	for (int level = 1; level <= PyrNum; ++level) {
		image_pyramid_impl(SrcImgData, srcWidth, srcHeight, level, PyrNum, OutImgData);
	}
}

void CShapeMatchV2::initial_shape_model_impl(
	ShapeModelV2& ModelID,
	const int Length,
	const int EdgeSize,
	const int PyrLevel)
{
	int AngleStep  = ModelID.m_AngleStep << PyrLevel;
	int AngleStart = ModelID.m_AngleStart;
	int AngleStop  = ModelID.m_AngleStop;

	if (ModelID.m_pShapeInfoPyr[PyrLevel] != nullptr) {
		std::free(ModelID.m_pShapeInfoPyr[PyrLevel]);
		ModelID.m_pShapeInfoPyr[PyrLevel] = nullptr;
	}

	if (PyrLevel != 0) {
		int AngleNum = 0;
		for (int iAngle = AngleStart; iAngle < AngleStop; iAngle += AngleStep) {
			if (iAngle == 0) continue;
			AngleNum++;
		}
		if (AngleStart == AngleStop) {
			AngleNum += 1;
		}
		else {
			AngleNum += 2;
		}

		ModelID.m_pShapeInfoPyr[PyrLevel] = (ShapeInfoV2*)malloc(AngleNum * sizeof(ShapeInfoV2));
		memset(ModelID.m_pShapeInfoPyr[PyrLevel], 0, AngleNum * sizeof(ShapeInfoV2));

		if (AngleStart <= 0) {
			ModelID.m_pShapeInfoPyr[PyrLevel][0].Angle = AngleStart;
			ModelID.m_pShapeInfoPyr[PyrLevel][AngleNum - 1].Angle = AngleStop;
		}
		else {
			ModelID.m_pShapeInfoPyr[PyrLevel][0].Angle = 0;
			ModelID.m_pShapeInfoPyr[PyrLevel][1].Angle = AngleStart;
			ModelID.m_pShapeInfoPyr[PyrLevel][AngleNum - 1].Angle = AngleStop;
		}
		bool isFilled = false;
		int Angle = AngleStart + AngleStep;
		for (int i = 1; i < AngleNum - 1; i++) {
			if ((!isFilled) && (Angle >= 0 && (Angle - AngleStep) < 0)) {
				ModelID.m_pShapeInfoPyr[PyrLevel][i].Angle = 0;
				if (Angle == 0) {
					Angle += AngleStep;
				}
				isFilled = true;
			}
			else if (Angle < AngleStop && Angle != 0) {
				ModelID.m_pShapeInfoPyr[PyrLevel][i].Angle = Angle;
				Angle += AngleStep;
			}
		}
		for (int i = 0; i < AngleNum; i++) {
			ModelID.m_pShapeInfoPyr[PyrLevel][i].PyLevel = PyrLevel;
			ModelID.m_pShapeInfoPyr[PyrLevel][i].AngleNum = AngleNum;
		}
	}
	else {
		int AngleNum = ModelID.m_AngleStop - ModelID.m_AngleStart + 1;
		ModelID.m_pShapeInfoPyr[PyrLevel] = (ShapeInfoV2*)malloc(AngleNum * sizeof(ShapeInfoV2));
		memset(ModelID.m_pShapeInfoPyr[PyrLevel], 0, AngleNum * sizeof(ShapeInfoV2));

		for (int i = 0; i < AngleNum; i++) {
			ModelID.m_pShapeInfoPyr[PyrLevel][i].PyLevel = PyrLevel;
			ModelID.m_pShapeInfoPyr[PyrLevel][i].Angle = AngleStart + i * ModelID.m_AngleStep;
			ModelID.m_pShapeInfoPyr[PyrLevel][i].AngleNum = AngleNum;
		}
	}
	
}

void CShapeMatchV2::initial_shape_model(
	ShapeModelV2& ModelID, 
	const int Width, 
	const int Height, 
	const int EdgeSize)
{
	ModelID.m_IsInited = true;
	int Length = ConvertLength(MAX(Width, Height));

	for (int level = 0; level <= ModelID.m_NumLevels; ++level) {
		ModelID.m_pShapeInfoPyr[level] = nullptr;
		initial_shape_model_impl(ModelID, Length, EdgeSize, level);
	}
}

void CShapeMatchV2::release_shape_model_impl(
	ShapeModelV2& ModelID,
	const int PyrLevel)
{
	ShapeInfoV2* pInfoPyr = nullptr;
	pInfoPyr = ModelID.m_pShapeInfoPyr[PyrLevel];
	std::free(pInfoPyr);
	pInfoPyr = nullptr;
}

bool CShapeMatchV2::release_shape_model(
	ShapeModelV2& ModelID)
{
	if (!ModelID.m_IsInited) {
		return false;
	}

	for (int level = 0; level <= ModelID.m_NumLevels; ++level) {
		release_shape_model_impl(ModelID, level);
	}

	ModelID.m_IsInited = false;
	return true;
}

void CShapeMatchV2::extract_shape_info(
	unsigned char* ImageData, 
	ShapeInfoV2* ShapeInfoData, 
	const int Contrast, 
	const int MinContrast,
	const int PointReduction, 
	unsigned char* MaskImgData)
{
	int width  = ShapeInfoData->ImgWidth;
	int height = ShapeInfoData->ImgHeight;
	uint32_t bufferSize  = width * height;

	unsigned char* pInput = (unsigned char*)malloc(bufferSize * sizeof(unsigned char));
	unsigned char* pBufOut = (unsigned char*)malloc(bufferSize * sizeof(unsigned char));
	int16_t	*pBufGradX = (int16_t *)malloc(bufferSize * sizeof(int16_t));
	int16_t	*pBufGradY = (int16_t *)malloc(bufferSize * sizeof(int16_t));
	int32_t	*pBufOrien = (int32_t *)malloc(bufferSize * sizeof(int32_t));
	float	*pBufMag = (float *)malloc(bufferSize * sizeof(float));

	if(pInput && pBufGradX && pBufGradY && pBufMag && pBufOrien && pBufOut) {
		//gaussian_filter(ImageData, pInput, width, height);
		memcpy(pInput, ImageData, bufferSize * sizeof(uint8_t));
		memset(pBufGradX,  0, bufferSize * sizeof(int16_t));
		memset(pBufGradY,  0, bufferSize * sizeof(int16_t));
		memset(pBufOrien,  -1, bufferSize * sizeof(int32_t));
		memset(pBufOut,    0, bufferSize * sizeof(uint8_t));
		memset(pBufMag,    0, bufferSize * sizeof(float));

		int index = 0;
		float MaxGradient = -9999.99f;

		for(int i = 1; i < width - 1; ++i) {
			for(int j = 1; j < height - 1; ++j) { 	
				index = j * width + i;
				if (*(MaskImgData + index) != 0xff) continue;

				int16_t sdx = *(pInput + index + 1) - *(pInput + index - 1);
				int16_t sdy = *(pInput + index + width) - *(pInput + index - width);
				float MagG = sqrt((float)(sdx*sdx) + (float)(sdy*sdy));

				*(pBufGradX + index) = sdx;
				*(pBufGradY + index) = sdy;
				*(pBufMag + index) = MagG;

				float direction = cv::fastAtan2(float(sdy), float(sdx));
				
				if ((direction > 0 && direction < 22.5) || (direction > 157.5 && direction < 202.5) || (direction > 337.5 && direction < 360))
					direction = 0;
				else if ((direction > 22.5 && direction < 67.5) || (direction > 202.5 && direction < 247.5))
					direction = 45;
				else if ((direction > 67.5 && direction < 112.5) || (direction > 247.5 && direction < 292.5))
					direction = 90;
				else if ((direction > 112.5 && direction < 157.5) || (direction > 292.5 && direction < 337.5))
					direction = 135;
				else
					direction = -1;

				pBufOrien[index] = (int32_t)direction;
				if (MagG > MaxGradient) MaxGradient = MagG;
			}
		}

		float factor = 255.0 / MaxGradient;
		float leftPixel = 0., rightPixel = 0.;
		for(int i = 1; i < width - 1; ++i) {
			for(int j = 1; j < height - 1; ++j) {
				index = j * width + i;
				switch (pBufOrien[index])
				{
				case 0: {
					leftPixel = *(pBufMag + index - 1);
					rightPixel = *(pBufMag + index + 1);
					break;
				}
				case 45: {
					leftPixel = *(pBufMag + index - width - 1);
					rightPixel = *(pBufMag + index + width + 1);
					break;
				}
				case 90: {
					leftPixel = *(pBufMag + index - width);
					rightPixel = *(pBufMag + index + width);
					break;
				}
				case 135: {
					leftPixel = *(pBufMag + index + width - 1);
					rightPixel = *(pBufMag + index - width + 1);
					break;
				}
				default:
					break;
				}
				
				if ((*(pBufMag + index) < leftPixel) ||
					(*(pBufMag + index) < rightPixel) || 
					(*(MaskImgData + index) == 0x00)){
					*(pBufOut + index) = 0;
				}
				else {
					*(pBufOut + index) = (unsigned char)(*(pBufMag + index)*factor);
				}
			}
		}

		int reserveFlag = 1, edgePointsCount = 0;
		for(int i = 1; i < width-1; i+= PointReduction) {
			for(int j = 1; j < height-1; j+= PointReduction) {
				index = j * width + i;
				int16_t fdx = *(pBufGradX + index);
				int16_t fdy = *(pBufGradY + index);
				int32_t dir = *(pBufOrien + index);
				float MagG = *(pBufMag + index);

				reserveFlag = 1;
				if((float)*(pBufOut + index) < Contrast) {
					if((float)*(pBufOut + index) < MinContrast) {
						*(pBufOut + index) = 0;
						reserveFlag = 0;
					}
					else
					{
						if (((float)*(pBufOut + index - width - 1) < Contrast) &&
							((float)*(pBufOut + index - width) < Contrast) &&
							((float)*(pBufOut + index - width + 1) < Contrast) &&
							((float)*(pBufOut + index - 1) < Contrast) &&
							((float)*(pBufOut + index + 1) < Contrast) &&
							((float)*(pBufOut + index + width - 1) < Contrast) &&
							((float)*(pBufOut + index + width) < Contrast) &&
							((float)*(pBufOut + index + width + 1) < Contrast))
						{
							*(pBufOut + index) = 0;
							reserveFlag = 0;
						}
					}
				}

				if(reserveFlag != 0) {
					if(fdx != 0 || fdy != 0) {		
						cv::Point pos(i - int(width*0.5), j - int(height*0.5));
						ShapeInfoData->Coordinates.push_back(pos);
						ShapeInfoData->EdgeDerivativeX.push_back(fdx);
						ShapeInfoData->EdgeDerivativeY.push_back(fdy);
						ShapeInfoData->EdgeDirection.push_back(dir);

						if (MagG != 0)
							ShapeInfoData->EdgeMagnitude.push_back(1 / MagG);
						else
							ShapeInfoData->EdgeMagnitude.push_back(0);
					}
				}
			}
		}

		edgePointsCount = ShapeInfoData->Coordinates.size();
		if (edgePointsCount != 0) {
			ShapeInfoData->NoOfCordinates = edgePointsCount;
			ShapeInfoData->ReferPoint.x = int(width*0.5);
			ShapeInfoData->ReferPoint.y = int(height*0.5);
		}
		else {
			std::cerr
				<< "extract_shape_info: no features."
				<< std::endl;
			return;
		}
	}

	if(pBufMag) std::free(pBufMag);
	if(pBufOrien) std::free(pBufOrien); 
	if(pBufGradY) std::free(pBufGradY);
	if(pBufGradX) std::free(pBufGradX);
	if(pBufOut) std::free(pBufOut);
	if(pInput) std::free(pInput);
}

bool CShapeMatchV2::build_model_list(
	ShapeInfoV2* ShapeInfoVec, 
	unsigned char* ImageData, 
	unsigned char* MaskData, 
	int Width, 
	int Height, 
	int Contrast, 
	int MinContrast, 
	int Granularity)
{
	int BufferSizeSrc = Width * Height;
	int tempLength = (int)(sqrt((float)BufferSizeSrc + (float)BufferSizeSrc) + 10);
	if ((tempLength & 1) != 0) tempLength += 1;

	int DstWidth = tempLength;
	int DstHeight = tempLength;
	int BufferSizeDst = DstWidth * DstHeight;

	unsigned char* pDstImgData = (unsigned char*)malloc(BufferSizeSrc * sizeof(unsigned char));
	memset(pDstImgData, 0, BufferSizeSrc * sizeof(unsigned char));

	unsigned char* pMaskRotData = (unsigned char*)malloc(BufferSizeSrc * sizeof(unsigned char));
	memset(pMaskRotData, 0, BufferSizeSrc * sizeof(unsigned char));

	cv::Mat SrcImage = cv::Mat(Height, Width, CV_8UC1);
	memcpy((unsigned char*)SrcImage.data, ImageData, BufferSizeSrc * sizeof(unsigned char));
	cv::Mat DstImage = cv::Mat(Height, Width, CV_8UC1);
	memcpy((unsigned char*)DstImage.data, pDstImgData, BufferSizeSrc * sizeof(unsigned char));

	cv::Mat SrcMask = cv::Mat(Height, Width, CV_8UC1);
	memcpy((unsigned char*)SrcMask.data, MaskData, BufferSizeSrc * sizeof(unsigned char));
	cv::Mat DstMask = cv::Mat(Height, Width, CV_8UC1);
	memcpy((unsigned char*)DstMask.data, pMaskRotData, BufferSizeSrc * sizeof(unsigned char));

	int AngleNum = ShapeInfoVec[0].AngleNum;
	for (int i = 0; i < AngleNum; i++)
	{
		ShapeInfoVec[i].ImgWidth  = DstImage.cols;
		ShapeInfoVec[i].ImgHeight = DstImage.rows;

		rotate_image(SrcImage, DstImage, ShapeInfoVec[i].Angle);
		rotate_image(SrcMask, DstMask, ShapeInfoVec[i].Angle);

		extract_shape_info(
			(unsigned char*)DstImage.data, 
			&ShapeInfoVec[i], 
			Contrast, 
			MinContrast, 
			Granularity, 
			(unsigned char*)DstMask.data
		);

		if (ShapeInfoVec[i].NoOfCordinates == 0) {
			std::cerr
				<< "build_model_list : no features."
				<< std::endl;
			return false;
		}
			
	}
	
	if(pMaskRotData) std::free(pMaskRotData);
	if(pDstImgData) std::free(pDstImgData);

	return true;
}

void CShapeMatchV2::train_shape_model(
	const cv::Mat& Image, 
	int Contrast, 
	int MinContrast,
	int PointReduction, 
	EdgeListV2& EdgeList)
{
	int width = Image.cols;
	int height = Image.rows;
	uint32_t  bufferSize = width * height;

	unsigned char* pInput = (unsigned char *)malloc(bufferSize * sizeof(unsigned char));
	unsigned char* pBufOut = (unsigned char *)malloc(bufferSize * sizeof(unsigned char));
	int16_t* pBufGradX = (int16_t *)malloc(bufferSize * sizeof(int16_t));
	int16_t* pBufGradY = (int16_t *)malloc(bufferSize * sizeof(int16_t));
	int32_t* pBufOrien = (int32_t *)malloc(bufferSize * sizeof(int32_t));
	float* pBufMag = (float *)malloc(bufferSize * sizeof(float));

	if( pInput && pBufGradX && pBufGradY && pBufMag && pBufOrien && pBufOut) {
		//gaussian_filter((unsigned char*)Image.data, pInput, width, height);
		memcpy(pInput, Image.data, sizeof(unsigned char)*bufferSize);
		memset(pBufGradX, 0, bufferSize * sizeof(int16_t));
		memset(pBufGradY, 0, bufferSize * sizeof(int16_t));
		memset(pBufOrien, -1, bufferSize * sizeof(int32_t));
		memset(pBufOut, 0, bufferSize * sizeof(uint8_t));
		memset(pBufMag, 0, bufferSize * sizeof(float));

		int index = 0;
		float MaxGradient = -9999.99f;
		for(int i = 1; i < width-1; ++i) {
			for(int j = 1; j < height-1; ++j){ 	
				index = j * width + i;
				int16_t sdx = *(pInput + index + 1) - *(pInput + index - 1);
				int16_t sdy = *(pInput + index + width) - *(pInput + index - width);
				float MagG = sqrt((float)(sdx*sdx) + (float)(sdy*sdy));

				*(pBufGradX + index) = sdx;
				*(pBufGradY + index) = sdy;
				*(pBufMag + index) = MagG;

				float direction = cv::fastAtan2(float(sdy), float(sdx));

				if ((direction > 0 && direction < 22.5) || (direction > 157.5 && direction < 202.5) || (direction > 337.5 && direction < 360))
					direction = 0;
				else if ((direction > 22.5 && direction < 67.5) || (direction > 202.5 && direction < 247.5))
					direction = 45;
				else if ((direction > 67.5 && direction < 112.5) || (direction > 247.5 && direction < 292.5))
					direction = 90;
				else if ((direction > 112.5 && direction < 157.5) || (direction > 292.5 && direction < 337.5))
					direction = 135;
				else
					direction = -1;

				pBufOrien[index] = (int32_t)direction;
				if (MagG > MaxGradient) MaxGradient = MagG;
			}
		}

		const float factor = 255.0 / MaxGradient;
		float leftPixel = 0., rightPixel = 0.;
		for(int i = 1; i < width - 1; ++i) {
			for(int j = 1; j < height - 1; ++j) {
				index = j * width + i;
				switch (pBufOrien[index]) {
				case 0: {
					leftPixel = *(pBufMag + index - 1);
					rightPixel = *(pBufMag + index + 1);
					break;
				}
				case 45: {
					leftPixel = *(pBufMag + index - width - 1);
					rightPixel = *(pBufMag + index + width + 1);
					break;
				}
				case 90: {
					leftPixel = *(pBufMag + index - width);
					rightPixel = *(pBufMag + index + width);
					break;
				}
				case 135: {
					leftPixel = *(pBufMag + index + width - 1);
					rightPixel = *(pBufMag + index - width + 1);
					break;
				}
				default:
					break;
				}

				float bufMag = *(pBufMag + index);
				if ((bufMag < leftPixel) || (bufMag < rightPixel))
					*(pBufOut + index) = 0;
				else
					*(pBufOut + index) = (unsigned char)(bufMag*factor);
			}
		}

		int flagReserve = 1, count = 0;
		for(int i = 1; i < width - 1; i += PointReduction) {
			for(int j = 1; j < height - 1; j += PointReduction) {
				index = j * width + i;
				int16_t fdx = *(pBufGradX + index);
				int16_t fdy = *(pBufGradY + index);
				float MagG = *(pBufMag + index);

				flagReserve = 1;
				if((float)*(pBufOut + index) < Contrast) {
					if((float)*(pBufOut + index) < MinContrast) {
						*(pBufOut + index) = 0;
						flagReserve = 0;
					}
					else {
						if (((float)*(pBufOut + index - width - 1) < Contrast) &&
							((float)*(pBufOut + index - width) < Contrast) &&
							((float)*(pBufOut + index - width + 1) < Contrast) &&
							((float)*(pBufOut + index - 1) < Contrast) &&
							((float)*(pBufOut + index + 1) < Contrast) &&
							((float)*(pBufOut + index + width - 1) < Contrast) &&
							((float)*(pBufOut + index + width) < Contrast) &&
							((float)*(pBufOut + index + width + 1) < Contrast)) {
							*(pBufOut + index) = 0;
							flagReserve = 0;
						}
					}
				}

				if(flagReserve != 0) {
					if (fdx != 0 || fdy != 0) {
						EdgeList.EdgePoint.push_back(cv::Point(i, j));
					}
				}
			}
		}
		EdgeList.ListSize = EdgeList.EdgePoint.size();
	}

	if(pBufMag) std::free(pBufMag);
	if(pBufOrien) std::free(pBufOrien);
	if(pBufGradY) std::free(pBufGradY);
	if(pBufGradX) std::free(pBufGradX);
	if(pBufOut) std::free(pBufOut);
	if(pInput) std::free(pInput);
}

bool CShapeMatchV2::create_shape_model_impl(
	ShapeModelV2& ModelID,
	const int BorderedWidth,
	const int BorderedHeight,
	unsigned char* pImageDataPyrAll,
	unsigned char* pMaskDataPyrAll,
	const int PyrLevel)
{
	int WidthPyr = BorderedWidth >> PyrLevel;
	int HeightPyr = BorderedHeight >> PyrLevel;
	int BufferSize = WidthPyr * HeightPyr;
	int in_size = BorderedWidth * BorderedHeight;
	int Contrast = ModelID.m_Contrast;
	int MinContrast = ModelID.m_MinContrast;
	int Granularity = ModelID.m_Granularity;

	int numerator = factorial(4, PyrLevel - 2);
	int denominator = std::powf(4, PyrLevel - 1);

	unsigned char* pImageDataPyr = (unsigned char*)malloc(BufferSize * sizeof(unsigned char));
	memcpy(pImageDataPyr, pImageDataPyrAll + (in_size/denominator)*numerator, BufferSize * sizeof(unsigned char));

	unsigned char* pMaskDataPyr = (unsigned char*)malloc(BufferSize * sizeof(unsigned char));
	memcpy(pMaskDataPyr, pMaskDataPyrAll + (in_size/denominator)*numerator, BufferSize * sizeof(unsigned char));

	cv::Mat cvImageDataPyr = cv::Mat(HeightPyr, WidthPyr, CV_8UC1, pImageDataPyr);
	cv::Mat cvMaskDataPyr = cv::Mat(HeightPyr, WidthPyr, CV_8UC1, pMaskDataPyr);

	bool IsBuild = build_model_list(
		ModelID.m_pShapeInfoPyr[PyrLevel],
		pImageDataPyr,
		pMaskDataPyr,
		WidthPyr,
		HeightPyr,
		Contrast,
		MinContrast,
		Granularity
	);

	return IsBuild;
}

bool CShapeMatchV2::create_shape_model(
	const cv::Mat& Template, 
	ShapeModelV2& ModelID)
{
	int ImgWidth = Template.cols;
	int ImgHeight = Template.rows;
	int PyrDepth = ModelID.m_NumLevels;

	if(PyrDepth >= 0) {
		int Length = ConvertLength(MAX(ImgWidth, ImgHeight));
		uint32_t  yOffset = (Length - ImgHeight) >> 1;
		uint32_t  xOffset = (Length - ImgWidth) >> 1;

		cv::Mat ImgBordered = cv::Mat(Length, Length, CV_8UC1);
		board_image(Template, ImgBordered, xOffset, yOffset);

		cv::Mat ImgMask = cv::Mat(Length, Length, CV_8UC1);
		memset(ImgMask.data, 0, ImgMask.rows * ImgMask.cols * sizeof(unsigned char));
		
		int ImgMaskWStep = ImgMask.cols;
		for (uint32_t row = yOffset; row < yOffset + Template.rows; row++)
			memset(ImgMask.data + row * ImgMaskWStep + xOffset, 0xff, Template.cols * sizeof(unsigned char));

		int BorderedWidth  = ImgBordered.cols;
		int BorderedHeight = ImgBordered.rows;

		bool IsBuild = false;
		int Contrast = ModelID.m_Contrast;
		int MinContrast = ModelID.m_MinContrast;
		int Granularity = ModelID.m_Granularity;
		
		int numerator = factorial(4, PyrDepth - 1);
		int denominator = std::powf(4, PyrDepth);
		uint32_t in_size = BorderedWidth * BorderedHeight;
		uint32_t out_size = (in_size / denominator)*numerator;

		unsigned char* pOut = (unsigned char*)malloc(out_size * sizeof(unsigned char));
		unsigned char* pOutMask = (unsigned char*)malloc(out_size * sizeof(unsigned char));

		image_pyramid(ImgBordered.data, BorderedWidth, BorderedHeight, PyrDepth, pOut);
		image_pyramid(ImgMask.data, BorderedWidth, BorderedHeight, PyrDepth, pOutMask);

		for (int level = 1; level <= ModelID.m_NumLevels; ++level) {
			IsBuild = create_shape_model_impl(
				ModelID,
				BorderedWidth,
				BorderedHeight,
				pOut,
				pOutMask,
				level
			);
			if (!IsBuild)
				return false;
		}

		IsBuild = build_model_list(
			ModelID.m_pShapeInfoPyr[0],
			(unsigned char*)(ImgBordered.data),
			(unsigned char*)(ImgMask.data),
			BorderedWidth,
			BorderedHeight,
			Contrast,
			MinContrast,
			Granularity
		);
		if (!IsBuild)
			return false;

		if (pOut) std::free(pOut);
		if (pOutMask) std::free(pOutMask);
	}
	ModelID.m_IsInited = true;
	return true;
}

void CShapeMatchV2::shape_match(
	unsigned char* SearchImage,
	ShapeInfoV2* ShapeInfoVec,
	const int Width, 
	const int Height, 
	int* NumMatches,
	const int Contrast, 
	const int MinContrast, 
	const float MinScore,
	const float Greediness, 
	SearchRegionV2* SearchRegion,
	MatchResultV2* ResultList)
{
	int width  = Width;
	int height = Height;
	uint32_t  bufferSize  = Width * Height;

	unsigned char* pInput = (unsigned char*)malloc(bufferSize * sizeof(unsigned char));
	int16_t* pBufGradX = (int16_t*)malloc(bufferSize * sizeof(int16_t));
	int16_t* pBufGradY = (int16_t*)malloc(bufferSize * sizeof(int16_t));
	float* pBufMag = (float*)malloc(bufferSize * sizeof(float));
	/*int16_t* pBufOrien = (int16_t*)malloc(bufferSize * sizeof(int16_t));*/

	resultsPerDeg.clear();
	totalResultsTemp.clear();
	resultsPerDeg.resize(MAXTARGETNUM);
	totalResultsTemp.resize(MAXTARGETNUM);

	if( pInput && pBufGradX && pBufGradY && pBufMag ){
		//gaussian_filter(SearchImage, pInput, width, height);
		memcpy(pInput, SearchImage, bufferSize * sizeof(uint8_t));
		memset(pBufGradX, 0, bufferSize * sizeof(int16_t));
		memset(pBufGradY, 0, bufferSize * sizeof(int16_t));
		memset(pBufMag, 0, bufferSize * sizeof(float));
		/*memset(pBufOrien, -1, bufferSize * sizeof(int16_t));*/

		int index = 0;
		float MaxGrad = -9999.0;

		for(int i = 1; i < width-1; ++i) {
			for(int j = 1; j < height-1; ++j) {
				index = j * width + i;
				int16_t sdx = *(pInput + index + 1) - *(pInput + index - 1);
				int16_t sdy = *(pInput + index + width) - *(pInput + index - width);
				*(pBufGradX + index) = sdx;
				*(pBufGradY + index) = sdy;
				*(pBufMag + index) = new_rsqrt((float)(sdx*sdx) + (float)(sdy*sdy));

				/*float direction = cv::fastAtan2(float(sdy), float(sdx));

				if ((direction > 0 && direction < 22.5) || (direction > 157.5 && direction < 202.5) || (direction > 337.5 && direction < 360))
					direction = 0;
				else if ((direction > 22.5 && direction < 67.5) || (direction > 202.5 && direction < 247.5))
					direction = 45;
				else if ((direction > 67.5 && direction < 112.5) || (direction > 247.5 && direction < 292.5))
					direction = 90;
				else if ((direction > 112.5 && direction < 157.5) || (direction > 292.5 && direction < 337.5))
					direction = 135;
				else
					direction = -1;

				pBufOrien[index] = (int16_t)direction;*/
			}
		}

		int curX = 0;
		int curY = 0;

		int16_t dir = 0;
		int16_t iTx = 0;
		int16_t iTy = 0;
		int16_t iSx = 0;
		int16_t iSy = 0;
		float   iSm = 0;
		float   iTm = 0;

		int startX =  SearchRegion->StartX;
		int startY =  SearchRegion->StartY;
		int endX   =  SearchRegion->EndX;
		int endY   =  SearchRegion->EndY;

		int AngleStop	= SearchRegion->AngleStop;
		int AngleStart	= SearchRegion->AngleStart;

		int	  PixelError   = 2;
		int	  AngleMatchNum = 0;
		int   SumOfCoords  = 0;
		int   TempPiontX   = 0;
		int   TempPiontY   = 0;
		float PartialSum   = 0;
		float PartialScore = 0;
		float ResultScore  = 0;
		float anMinScore = MinScore - 1.0;
		float NormMinScore   = 0;
		float NormGreediness = Greediness;

		int	resultsNumPerDegree = 0;
		int	totalResultsNum		= 0;

		for (int k = 0; k < ShapeInfoVec[0].AngleNum; ++k)
		{
			if (ShapeInfoVec[k].Angle < AngleStart || ShapeInfoVec[k].Angle > AngleStop) continue;

			resultsNumPerDegree = 0;
			ResultScore = 0;
			NormMinScore = MinScore / ShapeInfoVec[k].NoOfCordinates;
			NormGreediness = ((1 - Greediness * MinScore) / (1 - Greediness)) / ShapeInfoVec[k].NoOfCordinates;

			for(int i = startX; i < endX; ++i) {
				for(int j = startY; j < endY; ++j) {
					PartialSum = 0;
					AngleMatchNum = 0;
					for(int num = 0; num < ShapeInfoVec[k].NoOfCordinates; ++num) {
						curX = i + ShapeInfoVec[k].Coordinates[num].x;
						curY = j + ShapeInfoVec[k].Coordinates[num].y;
						iTx = ShapeInfoVec[k].EdgeDerivativeX[num];
						iTy = ShapeInfoVec[k].EdgeDerivativeY[num];
						iTm = ShapeInfoVec[k].EdgeMagnitude[num];
						dir = ShapeInfoVec[k].EdgeDirection[num];

						if (curX < 0 || curY < 0 || curX > width - 1 || curY > height - 1) continue;

						int offSet = curY * width + curX;
						iSx = *(pBufGradX + offSet);
						iSy = *(pBufGradY + offSet);
						iSm = *(pBufMag + offSet);

						/*if (dir != *(pBufOrien + offSet)) AngleMatchNum++;
						if (AngleMatchNum > int(ShapeInfoVec[k].NoOfCordinates*0.1)) break;*/

						if((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0)) {
							PartialSum = PartialSum + ((iSx * iTx) + (iSy * iTy)) * (iTm * iSm);
						}
						
						SumOfCoords = num + 1;
						PartialScore = PartialSum / SumOfCoords;
						if( PartialScore < (MIN(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
							break;
					}

					if (PartialScore > MinScore) {
						bool hasFlag = false;
						int Angle = ShapeInfoVec[k].Angle;
						auto resultPerDegTmp = resultsPerDeg.GetElement();
						auto resultPerDegSize = resultsPerDeg.size();
						for(int n = 1; n <= resultPerDegSize; ++n) {
							if(std::abs((resultPerDegTmp)[n].CenterLocX - i) < PixelError &&
								std::abs((resultPerDegTmp)[n].CenterLocY - j) < PixelError) {
								hasFlag = true;
								if((resultPerDegTmp)[n].ResultScore < PartialScore) {
									(resultPerDegTmp)[n].Angle = Angle;
									(resultPerDegTmp)[n].CenterLocX = i;
									(resultPerDegTmp)[n].CenterLocY = j;
									(resultPerDegTmp)[n].ResultScore = PartialScore;

									resultsPerDeg.clear();
									resultsPerDeg.resize(MAXTARGETNUM);
									for (int it = 1; it <= resultPerDegSize; ++it) {
										resultsPerDeg.push(resultPerDegTmp[it]);
									}

									break;
								}
							}
						}

						if(!hasFlag) {
							MatchResultV2 resultMatch;
							resultMatch.Angle = Angle;
							resultMatch.CenterLocX = i;
							resultMatch.CenterLocY = j;
							resultMatch.ResultScore = PartialScore;
							resultsPerDeg.push(resultMatch);
						}
					}
				}
			}

			auto resultPerDegTmp = resultsPerDeg.GetElement();
			for(int i = 1; i <= resultsPerDeg.size(); ++i) {
				totalResultsTemp.push((resultPerDegTmp)[i]);
			}
		}

		int resultsCounter = 0;
		bool hasFlag = false;
		auto totalResults = totalResultsTemp.GetElement();
		for(int i = 1; i <= totalResultsTemp.size(); i++) {
			hasFlag = false;
			for(int j = 0; j < resultsCounter; j++) {	
				if(std::abs((ResultList + j)->CenterLocX - (totalResults)[i].CenterLocX) < PixelError &&
					std::abs((ResultList + j)->CenterLocY - (totalResults)[i].CenterLocY) < PixelError) {
					hasFlag = true;
					if((totalResults)[i].ResultScore > (ResultList + j)->ResultScore) {
						(ResultList + j)->Angle			= (totalResults)[i].Angle;
						(ResultList + j)->CenterLocX	= (totalResults)[i].CenterLocX;
						(ResultList + j)->CenterLocY	= (totalResults)[i].CenterLocY;
						(ResultList + j)->ResultScore	= (totalResults)[i].ResultScore;
						break;
					}
				}
			}
			if(!hasFlag) {
				(ResultList + resultsCounter)->Angle = (totalResults)[i].Angle;
				(ResultList + resultsCounter)->CenterLocX = (totalResults)[i].CenterLocX;
				(ResultList + resultsCounter)->CenterLocY = (totalResults)[i].CenterLocY;
				(ResultList + resultsCounter)->ResultScore = (totalResults)[i].ResultScore;
				resultsCounter++;
			}
		}
		*NumMatches = resultsCounter;
	}

	if(pBufMag) std::free(pBufMag);
	if(pBufGradY) std::free(pBufGradY);
	if(pBufGradX) std::free(pBufGradX);
	if(pInput) std::free(pInput);
}

void CShapeMatchV2::shape_match_accurate(
	unsigned char* SearchImage, 
	ShapeInfoV2* ShapeInfoVec, 
	const int Width, 
	const int Height, 
	const int Contrast, 
	const int MinContrast, 
	float MinScore,
	float Greediness, 
	SearchRegionV2* SearchRegion,
	MatchResultV2* ResultList)
{
	int width  = Width;
	int height = Height;
	uint32_t  bufferSize  = Width * Height;

	unsigned char* pInput = (unsigned char*)malloc(bufferSize * sizeof(unsigned char));
	int16_t* pBufGradX   = (int16_t *) malloc(bufferSize * sizeof(int16_t));
	int16_t* pBufGradY   = (int16_t *) malloc(bufferSize * sizeof(int16_t));
	float* pBufMag     = (float *) malloc(bufferSize * sizeof(float));
	/*int16_t* pBufOrien = (int16_t*)malloc(bufferSize * sizeof(int16_t));*/

	if( pInput && pBufGradX && pBufGradY && pBufMag ) {
		//gaussian_filter(SearchImage, pInput, width, height);
		memcpy(pInput, SearchImage, bufferSize * sizeof(unsigned char));
		memset(pBufGradX, 0, bufferSize * sizeof(int16_t));
		memset(pBufGradY, 0, bufferSize * sizeof(int16_t));
		memset(pBufMag, 0, bufferSize * sizeof(float));
		/*memset(pBufOrien, -1, bufferSize * sizeof(int16_t));*/

		for(int i = 1; i < width-1; ++i) {
			for(int j = 1; j < height-1; ++j) {
				const int index = j * width + i;
				int16_t sdx = *(pInput + index + 1) - *(pInput + index - 1);
				int16_t sdy = *(pInput + index + width) - *(pInput + index - width);
				*(pBufGradX + index) = sdx;
				*(pBufGradY + index) = sdy; 
				*(pBufMag   + index) = new_rsqrt((float)(sdx*sdx) + (float)(sdy*sdy));

				/*float direction = cv::fastAtan2(float(sdy), float(sdx));

				if ((direction > 0 && direction < 22.5) || (direction > 157.5 && direction < 202.5) || (direction > 337.5 && direction < 360))
					direction = 0;
				else if ((direction > 22.5 && direction < 67.5) || (direction > 202.5 && direction < 247.5))
					direction = 45;
				else if ((direction > 67.5 && direction < 112.5) || (direction > 247.5 && direction < 292.5))
					direction = 90;
				else if ((direction > 112.5 && direction < 157.5) || (direction > 292.5 && direction < 337.5))
					direction = 135;
				else
					direction = -1;

				pBufOrien[index] = (int16_t)direction;*/
			}
		}

		int curX = 0;
		int curY = 0;

		int16_t dir = 0;
		int16_t iTx = 0;
		int16_t iTy = 0;
		int16_t iSx = 0;
		int16_t iSy = 0;
		float   iSm = 0;
		float   iTm = 0;

		int startX =  SearchRegion->StartX;
		int startY =  SearchRegion->StartY;
		int endX   =  SearchRegion->EndX;
		int endY   =  SearchRegion->EndY;

		int AngleStart	= SearchRegion->AngleStart;
		int AngleStop	= SearchRegion->AngleStop;

		int	  PixelError    = 2;
		int	  AngleMatchNum = 0;
		int	  ImageIndex	= 0;
		int   SumOfCoords	= 0;
		int   TempPointX	= 0;
		int   TempPointY	= 0;
		float PartialSum	= 0;
		float PartialScore	= 0;
		float ResultScore	= 0;
		float TempScore		= 0;
		float anMinScore = MinScore - 1.0;
		float NormMinScore	= 0;
		float NormGreediness= Greediness;

		for (int k = 0; k < ShapeInfoVec[0].AngleNum; ++k) {
			if (ShapeInfoVec[k].Angle < AngleStart || ShapeInfoVec[k].Angle > AngleStop) continue;

			ResultScore = 0;
			NormMinScore = MinScore / ShapeInfoVec[k].NoOfCordinates;
			NormGreediness = ((1- Greediness * MinScore)/(1-Greediness)) /ShapeInfoVec[k].NoOfCordinates;
			for(int i = startX; i < endX; ++i) {
				for(int j = startY; j < endY; ++j) {
					PartialSum = 0;
					AngleMatchNum = 0;
					for(int m = 0; m < ShapeInfoVec[k].NoOfCordinates; ++m) {
						curX = i + ShapeInfoVec[k].Coordinates[m].x ;
						curY = j + ShapeInfoVec[k].Coordinates[m].y ;
						iTx	 = ShapeInfoVec[k].EdgeDerivativeX[m];
						iTy	 = ShapeInfoVec[k].EdgeDerivativeY[m];
						iTm  = ShapeInfoVec[k].EdgeMagnitude[m];

						if(curX < 0 ||curY < 0||curX > width-1 ||curY > height-1) continue;

						ImageIndex = curY * width + curX;
						iSx = *(pBufGradX + ImageIndex);
						iSy = *(pBufGradY + ImageIndex);
						iSm = *(pBufMag   + ImageIndex);

						/*if (dir != *(pBufOrien + ImageIndex)) AngleMatchNum++;
						if (AngleMatchNum > int(ShapeInfoVec[k].NoOfCordinates*0.1)) break;*/

						if((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0)) {
							PartialSum = PartialSum + ((iSx * iTx) + (iSy * iTy)) * (iTm * iSm);
						}

						SumOfCoords = m + 1;
						PartialScore = PartialSum / SumOfCoords;
						if( PartialScore < (MIN(anMinScore + NormGreediness * SumOfCoords, NormMinScore * SumOfCoords)))
							break;
					}

					if(PartialScore > ResultScore) {
						ResultScore = PartialScore;
						TempPointX  = i;
						TempPointY  = j;
					}
				}
			}

			if (ResultScore > TempScore) {
				TempScore = ResultScore;
				ResultList->ResultScore = TempScore;
				ResultList->Angle = ShapeInfoVec[k].Angle;
				ResultList->CenterLocX = TempPointX;
				ResultList->CenterLocY = TempPointY;
			}
		}
	}

	if(pBufMag) std::free(pBufMag);
	if(pBufGradY) std::free(pBufGradY);
	if(pBufGradX) std::free(pBufGradX);
	if(pInput) std::free(pInput);
}

void CShapeMatchV2::find_shape_model_impl(
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
)
{
	int WidthPy = 0;
	int HeightPy = 0;
	float ScoreMax = 0;
	int MatchAngle = 0;
	int MatchPointX = 0;
	int MatchPointY = 0;
	int cropImgW = 0;
	int cropImgH = 0;
	int MatchNumCnt = 0;
	cv::Mat SearchImage, cropImage;
	int Contrast = ModelID.m_Contrast;
	int MinContrast = ModelID.m_MinContrast;
	int in_size = BorderedWidth * BorderedHeight;
	int Row1, Col1, Row2, Col2, ResultPointX, ResultPointY, ReferPointX, ReferPointY;
	int Offset = std::max(std::abs(ModelID.m_AngleStart), std::abs(ModelID.m_AngleStop));
	Offset = 0;

	SearchRegionV2* SearchRegion = (SearchRegionV2*)malloc(sizeof(SearchRegionV2));
	std::memset(SearchRegion, 0, sizeof(SearchRegionV2));
	PriorityQueue<MatchResultV2> MatchResult;
	MatchResult.resize(NumMatches);

	int factor = std::powf(4, PyrLevel);
	int numerator = factorial(4, PyrLevel - 2);
	int denominator = std::powf(4, PyrLevel - 1);

	unsigned char* pImagePyr = nullptr;
	pImagePyr = (unsigned char*)malloc((in_size/factor) * sizeof(unsigned char));

	if (PyrLevel == 0) {
		WidthPy = BorderedWidth;
		HeightPy = BorderedHeight;
		std::memcpy(pImagePyr, BorderedImgData, sizeof(unsigned char)*(in_size/factor));
	}
	else {
		WidthPy = BorderedWidth >> PyrLevel;
		HeightPy = BorderedHeight >> PyrLevel;
		std::memcpy(pImagePyr, pImagePyrAll + (in_size/denominator)*numerator, sizeof(unsigned char)*(in_size/factor));
	}
	
	SearchRegion->StartX = (ModelID.m_pShapeInfoPyr[PyrLevel][0].ReferPoint.x >> 1) + (xOffset >> PyrLevel);
	SearchRegion->StartY = (ModelID.m_pShapeInfoPyr[PyrLevel][0].ReferPoint.y >> 1) + (yOffset >> PyrLevel);
	SearchRegion->EndX = WidthPy - SearchRegion->StartX;
	SearchRegion->EndY = HeightPy - SearchRegion->StartY;
	SearchRegion->AngleRange = ModelID.m_pShapeInfoPyr[PyrLevel][0].AngleNum;
	SearchRegion->AngleStart = ModelID.m_AngleStart;
	SearchRegion->AngleStop = ModelID.m_AngleStop;
	SearchRegion->AngleStep = ModelID.m_AngleStep << PyrLevel;

	MatchResultV2 ResultListPyr[MAXTARGETNUM];
	std::memset(ResultListPyr, 0, MAXTARGETNUM * sizeof(MatchResultV2));

	cv::Mat cvImagePyr = cv::Mat(HeightPy, WidthPy, CV_8UC1, pImagePyr);

	if (ModelID.m_pShapeInfoPyr[PyrLevel] != NULL) {
		int TargetNum = 0;
		shape_match(
			pImagePyr,
			ModelID.m_pShapeInfoPyr[PyrLevel],
			WidthPy,
			HeightPy,
			&TargetNum,
			Contrast,
			MinContrast,
			MinScore,
			Greediness,
			SearchRegion,
			ResultListPyr
		);

		if (pImagePyr) {
			std::free(pImagePyr);
			pImagePyr = nullptr;
		}
	}
	else {
		if (pImagePyr) {
			std::free(pImagePyr);
			pImagePyr = nullptr;
		}

		std::cerr
			<< "level " << PyrLevel
			<< " search error."
			<< std::endl;
		return;
	}

	if (PyrLevel == 0) {
		std::memcpy(ResultList, ResultListPyr, NumMatches * sizeof(MatchResultV2));
		for (int it = 0; it < NumMatches; ++it) {
			ResultList[it].CenterLocX -= xOffset;
			ResultList[it].CenterLocY -= yOffset;
		}
		return;
	}

	for (int i = 0; i < MAXTARGETNUM; i++) {
		MatchPointX = ResultListPyr[i].CenterLocX;
		MatchPointY = ResultListPyr[i].CenterLocY;
		MatchAngle = ResultListPyr[i].Angle;
		ScoreMax = ResultListPyr[i].ResultScore;

		if (ScoreMax == 0) break;
		if (ScoreMax <= MinScore) continue;

		MatchResultV2 ResultPyr;
		for (int level = PyrLevel - 1; level >= 0; --level) {
			numerator = factorial(4, level - 2);
			denominator = std::powf(4, level - 1);

			factor = std::powf(4, level);
			pImagePyr = (unsigned char*)malloc((in_size / factor) * sizeof(unsigned char));

			if (level == 0) {
				WidthPy = BorderedWidth;
				HeightPy = BorderedHeight;
				std::memcpy(pImagePyr, BorderedImgData, in_size * sizeof(unsigned char));
			}
			else {
				WidthPy = BorderedWidth >> level;
				HeightPy = BorderedHeight >> level;
				std::memcpy(pImagePyr, pImagePyrAll + (in_size/denominator)*numerator, sizeof(unsigned char)*(in_size/factor));
			}

			ResultPointX = ((MatchPointX << 1) < 0) ? 0 : (MatchPointX << 1);
			ResultPointY = ((MatchPointY << 1) < 0) ? 0 : (MatchPointY << 1);

			ReferPointX = (ModelID.m_ImageWidth >> (level + 1));
			ReferPointY = (ModelID.m_ImageHeight >> (level + 1));

			int posOffset = 0;

			Row1 = ((ResultPointX - ReferPointX - 2 - posOffset) < 0) ? 0 : (ResultPointX - ReferPointX - 2 - posOffset);
			Col1 = ((ResultPointY - ReferPointY - 2 - posOffset) < 0) ? 0 : (ResultPointY - ReferPointY - 2 - posOffset);
			Row2 = ((ResultPointX + ReferPointX + 2 + posOffset) > WidthPy) ? WidthPy : (ResultPointX + ReferPointX + 2 + posOffset);
			Col2 = ((ResultPointY + ReferPointY + 2 + posOffset) > HeightPy) ? HeightPy : (ResultPointY + ReferPointY + 2 + posOffset);

			cropImgW = std::abs(Row1 - Row2);
			cropImgH = std::abs(Col1 - Col2);

			SearchImage = cv::Mat(HeightPy, WidthPy, CV_8UC1);
			memcpy((unsigned char*)SearchImage.data, pImagePyr, WidthPy*HeightPy * sizeof(unsigned char));
			cropImage = cv::Mat(cropImgH, cropImgW, CV_8UC1);
			gen_rectangle(SearchImage, cropImage, Row1, Col1);

			if (pImagePyr) {
				std::free(pImagePyr);
				pImagePyr = nullptr;
			}

			SearchRegion->StartX = ((ResultPointX - Row1 - 2 - posOffset) < 0) ? 0 : (ResultPointX - Row1 - 2 - posOffset);
			SearchRegion->StartY = ((ResultPointY - Col1 - 2 - posOffset) < 0) ? 0 : (ResultPointY - Col1 - 2 - posOffset);
			SearchRegion->EndX = ((SearchRegion->StartX + 4 + 2 * posOffset) > cropImgW) ? cropImgW : (SearchRegion->StartX + 4 + 2 * posOffset);
			SearchRegion->EndY = ((SearchRegion->StartY + 4 + 2 * posOffset) > cropImgH) ? cropImgH : (SearchRegion->StartY + 4 + 2 * posOffset);

			SearchRegion->AngleRange = ModelID.m_pShapeInfoPyr[level][0].AngleNum;
			//SearchRegion->AngleStart = ((MatchAngle - Offset) < ModelID.m_AngleStart) ? ModelID.m_AngleStart : (MatchAngle - Offset);
			//SearchRegion->AngleStop = ((MatchAngle + Offset) > ModelID.m_AngleStop) ? ModelID.m_AngleStop : (MatchAngle + Offset);
			SearchRegion->AngleStart = ModelID.m_AngleStart;
			SearchRegion->AngleStop = ModelID.m_AngleStop;
			SearchRegion->AngleStep = ModelID.m_AngleStep << level;

			if (ModelID.m_pShapeInfoPyr[level] != NULL) {
				shape_match_accurate(
					(unsigned char*)cropImage.data,
					ModelID.m_pShapeInfoPyr[level],
					cropImgW,
					cropImgH,
					Contrast,
					MinContrast,
					MinScore,
					Greediness,
					SearchRegion,
					&ResultPyr);
			}
			else {
				std::cerr
					<< "level " << level
					<< " search error."
					<< std::endl;
				return;
			}

			MatchPointX = ResultPyr.CenterLocX + Row1;
			MatchPointY = ResultPyr.CenterLocY + Col1;
			MatchAngle = ResultPyr.Angle;
			ScoreMax = ResultPyr.ResultScore;
			if (ScoreMax < MinScore) break;
		}

		if (ResultPyr.ResultScore > MinScore) {
			ResultPyr.CenterLocX = ResultPyr.CenterLocX + Row1 - xOffset;
			ResultPyr.CenterLocY = ResultPyr.CenterLocY + Col1 - yOffset;
			MatchResult.push(ResultPyr);
		}
	}

	auto MatchResultEle = MatchResult.GetElement();
	for (int it = 1; it <= MatchResult.size(); ++it) {
		ResultList[MatchNumCnt] = MatchResultEle[it];
		MatchNumCnt++;
	}
}

void CShapeMatchV2::find_shape_model(
	const cv::Mat& Image, 
	ShapeModelV2& ModelID, 
	const float MinScore, 
	const int NumMatches, 
	const float Greediness, 
	MatchResultV2* ResultList)
{
	int ImgWidth  = Image.cols;
	int ImgHeight = Image.rows;
	int PyrDepth = ModelID.m_NumLevels;

	if(PyrDepth >= 0) {
		int width = 0;
		int height = 0;
		int xOffset = 0;
		int yOffset = 0;
		uint8_t *pData = NULL;
		bool isBordred = false;
		cv::Mat ImgBordered;

		if((ImgWidth % 16 != 0 ) && (ImgHeight % 16 != 0)) {
			int BorderedWidth  = ConvertLength(ImgWidth);
			int BorderedHeight = ConvertLength(ImgHeight);

			xOffset = (BorderedWidth - ImgWidth) >> 1;
			yOffset = (BorderedHeight - ImgHeight) >> 1;

			ImgBordered = cv::Mat(BorderedHeight, BorderedWidth, CV_8UC1);
			board_image(Image, ImgBordered, xOffset, yOffset);
			isBordred = true;
			width  = BorderedWidth;
			height = BorderedHeight;
			pData = (unsigned char*)ImgBordered.data;

#ifdef VISUALIZATION
			cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
			cvNamedWindow("Bounded", CV_WINDOW_AUTOSIZE);
			cvShowImage("Original", Image);
			cvShowImage("Bounded", ImgBordered);
			cvWaitKey(0);
#endif
		} else {
			width  = ImgWidth;
			height = ImgHeight;
			pData = (unsigned char*)Image.data;
		}

		int numerator = factorial(4, PyrDepth - 1);
		int denominator = std::powf(4, PyrDepth);

		uint32_t in_size = width * height;
		uint32_t out_size = (in_size/denominator)*numerator;

		unsigned char* pOut = (unsigned char*)malloc(out_size * sizeof(unsigned char));
		image_pyramid(pData, width, height, PyrDepth, pOut);

		PriorityQueue<MatchResultV2> MatchResult;
		MatchResult.resize(NumMatches);

		find_shape_model_impl(
			ModelID,
			PyrDepth,
			NumMatches,
			width,
			height,
			pData,
			xOffset,
			yOffset,
			MinScore,
			Greediness,
			pOut,
			ResultList
		);

		if (pOut) {
			std::free(pOut);
			pOut = nullptr;
		}
	}
	else {
		std::cerr
			<< "find_shape_model: illegal input."
			<< std::endl;
		return;
	}
}

int ShapeMatchTool::ShiftCos(int y)
{
	if (y<0) y*=-1;
	y %= 360;
	if ( y > 270 )
	{
		return ShiftCos((360 - y));
	}
	else if ( y > 180 )
	{
		return - ShiftCos((y - 180));
	}
	else if ( y > 90 )
	{
		return - ShiftCos((180 - y));
	}
	int index  = (y >> 2);
	int offset = (y % 4);
	// on the borderline of overflowing if use JInt16
	int cosVal = (4 - offset) * K_CosineTable[index]
	+ offset * K_CosineTable[index + 1];
	return cosVal >> 2;
}

int ShapeMatchTool::ShiftSin(int y)
{
	return ShiftCos(y + 270);
}

float ShapeMatchTool::Q_rsqrt(float number)
{
	//long i;
	//float x2, y;
	//const float threehalfs = 1.5F;
	//x2 = number * 0.5F;
	//y  = number;
	//i  = * ( long * ) &y;  // evil floating point bit level hacking
	//i  = 0x5f3759df - ( i >> 1 );
	//y  = * ( float * ) &i;
	//y  = y * ( threehalfs - ( x2 * y * y ) ); // 1st iteration
	//y  = y * ( threehalfs - ( x2 * y * y ) ); // 2nd iteration, this can be removed
	//return y;

	float xhalf = 0.5f*number;
	int i = *(int*)&number; // get bits for floating VALUE
	i = 0x5f375a86- (i>>1); // gives initial guess y0
	number = *(float*)&i; // convert bits BACK to float
	number = number*(1.5f-xhalf*number*number); // Newton step, repeating increases accuracy
	return number;
}

float ShapeMatchTool::new_rsqrt(float f)
{
	//这是调用用CPU SSE指令集中rsqrt函数直接得出结果

	//__m128 m_a = _mm_set_ps1(f);
	//__m128 m_b = _mm_rsqrt_ps(m_a);

	//return m_b[0];

	return 1/sqrtf(f);
}

void ShapeMatchTool::QuickSort(MatchResultV2 *s, int l, int r)
{
	int i, j;
	MatchResultV2 Temp;
	Temp.Angle 		 = 0;
	Temp.CenterLocX  = 0;
	Temp.CenterLocY  = 0;
	Temp.ResultScore = 0;
	if (l < r)
	{
		i = l;
		j = r;
		Temp = s[i];
		while (i < j)
		{
			while(i < j && s[j].ResultScore < Temp.ResultScore)	//修改符号可以完成升序或者降序
				j--;
			if(i < j)
				s[i++] = s[j];

			while(i < j && s[i].ResultScore > Temp.ResultScore)	//修改符号可以完成升序或者降序
				i++;
			if(i < j)
				s[j--] = s[i];
		}
		s[i] = Temp;
		QuickSort(s, l, i-1);
		QuickSort(s, i+1, r);
	}
}

int ShapeMatchTool::ConvertLength(int LengthSrc)
{
	int temp = 0;
	for (int i = 4; ; i++)
	{
		temp = (int)pow(2, i);
		if (temp >= LengthSrc)
		{
			LengthSrc = temp;
			break;
		}
	}
	return LengthSrc;
}

int ShapeMatchTool::factorial(int num, int level)
{
	if (level < 0) return 0;

	int sum = 0;
	for (int it = 0; it <= level; ++it) {
		sum += std::powf(num, it);
	}
	return sum;
}

/* 
 Various border types, image boundaries are denoted with '|' 
  
 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh 
 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb 
 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba 
 * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg         
 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i' 
 */  
int cv::borderInterpolate( int p, int len, int borderType ) // p是扩展边界的位置，len是原图宽度  
{  
    if( (unsigned)p < (unsigned)len )     // 转换为无符号类型，左边界和上边界：p一般是负数，右边界和下边界，p一般是大于len的。  
        ;  
    else if( borderType == BORDER_REPLICATE ) // 重复类型，每次对应原图的位置是0或len-1  
        p = p < 0 ? 0 : len - 1;  
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 ) // 反射/映射  
    {  
        int delta = borderType == BORDER_REFLECT_101;  
        if( len == 1 )  
            return 0;  
        do  
        {  
            if( p < 0 )    // 反射：左边界或101：右边界  
                p = -p - 1 + delta;  
            else  
                p = len - 1 - (p - len) - delta;  
        }  
        while( (unsigned)p >= (unsigned)len );  
    }  
    else if( borderType == BORDER_WRAP )  // 包装  
    {  
        if( p < 0 )  // 左边界  
            p -= ((p-len+1)/len)*len;  
        if( p >= len )  // 右边界  
            p %= len;  
    }  
    else if( borderType == BORDER_CONSTANT )  // 常量，另外处理  
        p = -1;  
    else  
        //CV_Error( CV_StsBadArg, "Unknown/unsupported border type" );  
    return p;  
}

static void copyMakeBorder_8u( const uchar* src, size_t srcstep, cv::Size srcroi, // 原图 参数：数据，step，大小
							  uchar* dst, size_t dststep, cv::Size dstroi,  // 目的图像参数
							  int top, int left, int cn, int borderType )
{
	const int isz = (int)sizeof(int);
	int i, j, k, elemSize = 1;
	bool intMode = false;

	if( (cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0 )
	{
		cn /= isz;
		elemSize = isz;
		intMode = true;
	}

	cv::AutoBuffer<int> _tab((dstroi.width - srcroi.width)*cn);  // 大小是扩展的左右边界之和，仅用于存放扩展的边界在原图中的位置
	int* tab = _tab;
	int right = dstroi.width - srcroi.width - left;
	int bottom = dstroi.height - srcroi.height - top;

	for( i = 0; i < left; i++ ) // 左边界
	{
		j = cv::borderInterpolate(i - left, srcroi.width, borderType)*cn;  // 计算出原图中对应的位置
		for( k = 0; k < cn; k++ )  // 每个通道的处理
			tab[i*cn + k] = j + k;
	}

	for( i = 0; i < right; i++ )  // 右边界
	{
		j = cv::borderInterpolate(srcroi.width + i, srcroi.width, borderType)*cn;
		for( k = 0; k < cn; k++ )
			tab[(i+left)*cn + k] = j + k;
	}

	srcroi.width *= cn;
	dstroi.width *= cn;
	left *= cn;
	right *= cn;

	uchar* dstInner = dst + dststep*top + left*elemSize;

	for( i = 0; i < srcroi.height; i++, dstInner += dststep, src += srcstep ) // 从原图中复制数据到扩展的边界中
	{
		if( dstInner != src )
			memcpy(dstInner, src, srcroi.width*elemSize);

		if( intMode )
		{
			const int* isrc = (int*)src;
			int* idstInner = (int*)dstInner;
			for( j = 0; j < left; j++ )
				idstInner[j - left] = isrc[tab[j]];
			for( j = 0; j < right; j++ )
				idstInner[j + srcroi.width] = isrc[tab[j + left]];
		}
		else
		{
			for( j = 0; j < left; j++ )
				dstInner[j - left] = src[tab[j]];
			for( j = 0; j < right; j++ )
				dstInner[j + srcroi.width] = src[tab[j + left]];
		}
	}

	dstroi.width *= elemSize;
	dst += dststep*top;

	for( i = 0; i < top; i++ )  // 上边界
	{
		j = cv::borderInterpolate(i - top, srcroi.height, borderType);
		memcpy(dst + (i - top)*dststep, dst + j*dststep, dstroi.width); // 进行整行的复制
	}

	for( i = 0; i < bottom; i++ ) // 先边界
	{
		j = cv::borderInterpolate(i + srcroi.height, srcroi.height, borderType);
		memcpy(dst + (i + srcroi.height)*dststep, dst + j*dststep, dstroi.width); // 进行整行的复制
	}
}