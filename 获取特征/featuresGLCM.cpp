#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include "featuresGLCM.h"


using namespace std;
using namespace cv;

/************************************************************************/
/* ��ȡ��������: ���������ء����Ծء���صľ�ֵ�ͱ�׼����Ϊ����8ά��������*/
/************************************************************************/


void calGLCMNormalization(Mat M, double *vectorGLCM)
{
	imwrite("temp_01.jpg", M);
	IplImage *img_rgb = cvLoadImage("temp_01.jpg");

	IplImage* img_gray = cvCreateImage(cvGetSize(img_rgb), 8, 1);
	cvCvtColor(img_rgb, img_gray, CV_BGR2GRAY);
	CvMat *mat_img;
	mat_img = cvCreateMat(img_gray->height, img_gray->width, \
		CV_64FC1);
	cvConvert(img_gray, mat_img);
	//double vectorGLCM[8];
	GetGLCM(mat_img, vectorGLCM);

	//cvWaitKey(0);
	cvReleaseImage(&img_rgb);
	cvReleaseImage(&img_gray);
	cvReleaseMat(&mat_img);
}

int GetGLCM(CvMat* mat_low_double, double *glcm)
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
	double localSteady[4] = { 0 };//�ֲ�ƽ����

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
				localSteady[dim] = temp/(1 + (m - n)*(m - n)) + localSteady[dim];//�ֲ�ƽ����
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
	glcm[8] = GetMean(localSteady);
	glcm[9] = GetSdDe(localSteady, glcm[8]);

	//��ʾ��ȡ��ȫ����������  
// 	cout << "�������ء����Ծء���صľ�ֵ�ͱ�׼����Ϊ����8ά��������" << endl;
// 	for each (double var in glcm)
// 	{
// 		cout << var << endl;
// 	}

	return 0;
}


/************************************************************************/
/* ��ȡƽ��ֵ                                                                     */
/************************************************************************/
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

