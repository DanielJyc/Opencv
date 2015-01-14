#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
using namespace std;
using namespace cv;

void calGLCMNormalization(Mat M, double *vectorGLCM);
int GetGLCM(CvMat* mat_low_double, double *glcm);
double GetMean(double m[4]);
double GetSdDe(double m[4], double nmean);

