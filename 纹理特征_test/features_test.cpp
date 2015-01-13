// main.cpp : �������̨Ӧ�ó������ڵ㡣
//

//#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

using namespace std;
using namespace cv;

#define GLCM_DIS 3  //�Ҷȹ��������ͳ�ƾ���
#define GLCM_CLASS 16 //����Ҷȹ��������ͼ��Ҷ�ֵ�ȼ���
#define GLCM_ANGLE_HORIZATION 0  //ˮƽ
#define GLCM_ANGLE_VERTICAL   1	 //��ֱ
#define GLCM_ANGLE_DIGONAL    2  //�Խ�

int GetGLCM(CvMat* mat_low, double[8]);
double GetMean(double[4]);
double GetSdDe(double[4], double);

/************************************************************************/
/*  �ο� http://blog.csdn.net/cxf7394373/article/details/6988229         */
/************************************************************************/
int calGLCM(IplImage* bWavelet, int angleDirection, double* featureVector)
{
	int i, j;
	int width, height;

	if (NULL == bWavelet)
		return 1;

	width = bWavelet->width;
	height = bWavelet->height;

	int * glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];

	if (NULL == glcm || NULL == histImage)
		return 2;

	//�Ҷȵȼ���---��GLCM_CLASS���ȼ�
	uchar *data = (uchar*)bWavelet->imageData;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			histImage[i * width + j] = (int)(data[bWavelet->widthStep * i + j] * GLCM_CLASS / 256);
		}
	}

	//��ʼ����������
	for (i = 0; i < GLCM_CLASS; i++)
	for (j = 0; j < GLCM_CLASS; j++)
		glcm[i * GLCM_CLASS + j] = 0;

	//����Ҷȹ�������
	int w, k, l;
	//ˮƽ����
	if (angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//��ֱ����
	else if (angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//�ԽǷ���
	else if (angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];

				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	//��������ֵ
	double entropy = 0, energy = 0, contrast = 0, homogenity = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	{
		for (j = 0; j < GLCM_CLASS; j++)
		{
			//��
			if (glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//����
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//�Աȶ�
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//һ����
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//��������ֵ
	i = 0;
	featureVector[i++] = entropy;
	featureVector[i++] = energy;
	featureVector[i++] = contrast;
	featureVector[i++] = homogenity;
	for (int j = 0; j < 4;j++)
	{
		cout << featureVector[j] << endl;
	}

	delete[] glcm;
	delete[] histImage;

	return 0;
}


void calGLCMNormalization(IplImage* img_rgb)
{
	IplImage* img_gray = cvCreateImage(cvGetSize(img_rgb), 8, 1);
	cvCvtColor(img_rgb, img_gray, CV_BGR2GRAY);
	CvMat *mat_img;
	mat_img = cvCreateMat(img_gray->height, img_gray->width, \
		CV_64FC1);
	cvConvert(img_gray, mat_img);
	double vectorGLCM[8];
	GetGLCM(mat_img, vectorGLCM);

	cvWaitKey(0);
	cvReleaseImage(&img_rgb);
	cvReleaseImage(&img_gray);
	cvReleaseMat(&mat_img);
}

int GetGLCM(CvMat* mat_low_double, double glcm[8])
{
	double minvalue = 0.0, maxvalue = 0.0;
	int nW, nH;
	int j, i, n, m, dim;
	nW = mat_low_double->width;
	nH = mat_low_double->height;
	CvMat *mat_low_uchar;
	mat_low_uchar = cvCreateMat(nH, nW, CV_8UC1);
	cvMinMaxLoc(mat_low_double, &minvalue, &maxvalue);
	cvConvertScale(mat_low_double, mat_low_double, 1, -minvalue);
	cvConvertScale(mat_low_double, mat_low_double, 255 / (maxvalue - minvalue), 0);
	cvConvert(mat_low_double, mat_low_uchar);

	/*--------------------------------------------------------------------------
	2.Ϊ�˼��ټ���������ԭʼͼ��Ҷȼ�ѹ������Gray������16��
	�㷨ԭ����255��Ϊ16�ݣ�gray(i,j)���ڣ�0,15��Ϊ1��gray(i,j)���ڣ�16,31��Ϊ2
	--------------------------------------------------------------------------*/
	for (j = 0; j<nH; j++)
	for (i = 0; i<nW; i++)
	for (n = 0; n<16; n++)
	{
		uchar temp = CV_MAT_ELEM(*mat_low_uchar, uchar, j, i);
		if ((n * 16 <= temp) && (temp <= n * 16 + 15))
			*((uchar*)CV_MAT_ELEM_PTR(*mat_low_uchar, j, i)) = n;
	}

	/*--------------------------------------------------------------------------
	3.�����ĸ���������P,ȡ����Ϊ1���Ƕȷֱ�Ϊ0,45,90,135
	--------------------------------------------------------------------------*/
	CvMat* mat_co[4];
	for (dim = 0; dim<4; dim++)
	{
		mat_co[dim] = cvCreateMat(16, 16, CV_64FC1);
		cvZero(mat_co[dim]);
	}

	for (m = 0; m<16; m++)//�Ҷȼ�
	{
		for (n = 0; n<16; n++)
		{
			for (j = 0; j<nH; j++)//ͼ���С�ߴ�
			{
				for (i = 0; i<nW; i++)
				{
					uchar temp = CV_MAT_ELEM(*mat_low_uchar, uchar, j, i);
					double tempmatvalue;

					if (i<(nW - 1) && temp == m && (CV_MAT_ELEM(*mat_low_uchar, uchar, j, i + 1) == n))
					{
						tempmatvalue = cvGetReal2D(mat_co[0], m, n);
						cvSetReal2D(mat_co[0], m, n, (tempmatvalue + 1.0));
						cvSetReal2D(mat_co[0], n, m, (tempmatvalue + 1.0));
					}
					if (j>0 && i<(nW - 1) && temp == m && (CV_MAT_ELEM(*mat_low_uchar, uchar, j - 1, i + 1)) == n)
					{
						tempmatvalue = cvGetReal2D(mat_co[1], m, n);
						cvSetReal2D(mat_co[1], m, n, (tempmatvalue + 1.0));
						cvSetReal2D(mat_co[1], n, m, (tempmatvalue + 1.0));
					}
					if (j<(nH - 1) && temp == m && (CV_MAT_ELEM(*mat_low_uchar, uchar, j + 1, i)) == n)
					{
						tempmatvalue = cvGetReal2D(mat_co[2], m, n);
						cvSetReal2D(mat_co[2], m, n, (tempmatvalue + 1.0));
						cvSetReal2D(mat_co[2], n, m, (tempmatvalue + 1.0));
					}
					if (j<(nH - 1) && i<(nW - 1) && temp == m && (CV_MAT_ELEM(*mat_low_uchar, uchar, j + 1, i + 1)) == n)
					{
						tempmatvalue = cvGetReal2D(mat_co[3], m, n);
						cvSetReal2D(mat_co[3], m, n, (tempmatvalue + 1.0));
						cvSetReal2D(mat_co[3], n, m, (tempmatvalue + 1.0));
					}
				}
			}
			if (m == n)
			for (dim = 0; dim<4; dim++)
			{
				double tempmatvalue1 = cvGetReal2D(mat_co[dim], m, n);
				cvSetReal2D(mat_co[dim], m, n, (tempmatvalue1 * 2));
			}
		}
	}

	/*--------------------------------------------------------------
	�Թ��������һ��
	---------------------------------------------------------------*/
	CvMat *temp_mat = cvCreateMat(16, 16, CV_64FC1);
	for (dim = 0; dim<4; dim++)
	{
		CvScalar sum_value = cvSum(mat_co[dim]);
		// 		sum_value.val[0] = 1/sum_value.val[0];
		// 		sum_value.val[1] = sum_value.val[2] = sum_value.val[3] = sum_value.val[0];
		cvSet(temp_mat, sum_value);
		//cvScaleAdd(mat_co[dim],sum_value,NULL,mat_co[dim]);
		cvDiv(mat_co[dim], temp_mat, mat_co[dim]);
	}

	/*--------------------------------------------------------------------------
	4.�Թ�����������������ء����Ծء����4���������
	--------------------------------------------------------------------------*/
	double energy[4] = { 0 }; //����
	double entropy[4] = { 0 }; //��
	double inertia[4] = { 0 };//���Ծ�
	double uX[4] = { 0 };/*������Ц�x*/  double uY[4] = { 0 };/*������Ц�y*/
	double deltaX[4] = { 0 };/*������Ц�x*/ double deltaY[4] = { 0 };/*������Ц�y*/
	double corelation[4] = { 0 }; //�����

	CvScalar temp_energy[4];  //����
	CvMat *mat_energy[4];
	for (dim = 0; dim<4; dim++)
	{
		mat_energy[dim] = cvCreateMat(16, 16, CV_64FC1);
		cvMul(mat_co[dim], mat_co[dim], mat_energy[dim]);
		temp_energy[dim] = cvSum(mat_energy[dim]);   //������
		energy[dim] = temp_energy[dim].val[0];
	}

	for (dim = 0; dim<4; dim++)
	{
		for (m = 0; m<16; m++)
		{
			for (n = 0; n<16; n++)
			{
				double temp = cvGetReal2D(mat_co[dim], m, n);
				if (temp != 0)
				{
					entropy[dim] = temp*log(temp) + entropy[dim]; //��
				}
				inertia[dim] = (m - n)*(m - n)*temp + inertia[dim];  //���Ծ�
				uX[dim] = n*temp + uX[dim]; //������Ц�x
				uY[dim] = n*temp + uY[dim]; //������Ц�y
			}
		}
	}
	for (dim = 0; dim<4; dim++)
	{
		for (m = 0; m<16; m++)
		{
			for (n = 0; n<16; n++)
			{
				double temp = cvGetReal2D(mat_co[dim], m, n);

				deltaX[dim] = (m - uX[dim])*(m - uX[dim])*temp + deltaX[dim]; //������Ц�x
				deltaY[dim] = (n - uY[dim])*(n - uY[dim])*temp + deltaY[dim]; //������Ц�y
				corelation[dim] = m*n*temp + corelation[dim];
			}
		}
		corelation[dim] = (corelation[dim] - uX[dim] * uY[dim]) / deltaX[dim] / deltaY[dim]; //�����   
	}

	/*--------------------------------------------------------------------------
	5.���������ء����Ծء���صľ�ֵ�ͱ�׼����Ϊ����8ά��������
	---------------------------------------------------------------------------*/
	glcm[0] = GetMean(energy);
	glcm[1] = GetSdDe(energy, glcm[0]);
	glcm[2] = GetMean(entropy);
	glcm[3] = GetSdDe(entropy, glcm[2]);
	glcm[4] = GetMean(inertia);
	glcm[5] = GetSdDe(inertia, glcm[4]);
	glcm[6] = GetMean(corelation);
	glcm[7] = GetSdDe(corelation, glcm[6]);

	cout << "energy" << endl;
	for each (double var in energy)
	{
		cout << var << endl;
	}
	cout << "entropy" << endl;
	for each (double var in entropy)
	{
		cout << var << endl;
	}
	cout << "inertia" << endl;
	for each (double var in inertia)
	{
		cout << var << endl;
	}
	cout << "corelation" << endl;
	for each (double var in corelation)
	{
		cout << var << endl;
	}
	cout << "�������ء����Ծء���صľ�ֵ�ͱ�׼����Ϊ����8ά��������" << endl;
	for each (double var in glcm)
	{
		cout << var << endl;
	}

	cvNamedWindow("test");
	cvShowImage("test", mat_low_uchar);
	cvWaitKey(0);
	return 0;
}

double GetMean(double m[4])
{
	double nmean = 0, sum = 0;
	sum = m[0] + m[1] + m[2] + m[3];
	nmean = sum / 4;
	return nmean;
}

double GetSdDe(double m[4], double nmean)
{
	int i;
	double stde, sum = 0;
	for (i = 0; i<4; i++)
	{
		sum += ((m[i] - nmean)*(m[i] - nmean));
	}
	sum /= 4;
	stde = sqrt(sum);
	return stde;
}

void getGLCMFeatures(Mat M)
{
	IplImage *img = &IplImage(M);
	cvNamedWindow("test");
	cvShowImage("test", img);
	calGLCMNormalization(img); //��� ��һ��

}

int main()
{
 //	Mat M = imread("01.jpg");
// 	Mat srcImg;                         // Mat type variable .
// 	IplImage *resIplPtr = NULL;         // Initialize by NULL.
// 	srcImg = imread("01.jpg");         // read image;         
// 	resIplPtr = &(IplImage(srcImg));    // Mat to IplImage Pointer 
// 	calGLCMNormalization(resIplPtr); //��� ��һ��
// 	resIplPtr = NULL;                   // set as NULL.

	Mat M1 = imread("01.jpg");
	imwrite("001.jpg", M1);
	IplImage* img_rgb = cvLoadImage("001.jpg");
	calGLCMNormalization(img_rgb); //��� ��һ��


// 	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\�����������ڰ׻�\\p4.jpg");
// 
	//getGLCMFeatures(srcImg);

// 	IplImage *img = &IplImage(M);
// 	cvNamedWindow("test");
// 	cvShowImage("test", img);
// 	calGLCMNormalization(img); //��� ��һ��

	//IplImage* img_rgb = cvLoadImage("01.jpg");
// 	IplImage* img_rgb = cvLoadImage("6330.jpg");
// 
// 	calGLCMNormalization(img_rgb); //��� ��һ��

// 	double featureVector[4];  
// 	calGLCM(img_rgb, 0, featureVector);//δ��һ��




	waitKey();
	return 0;
}