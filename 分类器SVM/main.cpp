#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace std;
/************************************************************************/
/* ���Դ��룬�ٷ����̣�
˵��:http://blog.csdn.net/yang_xian521/article/details/6969904
h�ͣ�*/
/************************************************************************/
void exampleAboutImg()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[5] = { 1.0, -1.0, 1.0, -1.0, 1.0 };
	Mat labelsMat(5, 1, CV_32FC1, labels);


	float trainingData[5][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 }, { 501, 128 } };
	Mat trainingDataMat(5, 2, CV_32FC1, trainingData);

	//����֧���������Ĳ���
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;//SVM���ͣ�ʹ��C֧��������
	params.kernel_type = CvSVM::LINEAR;//�˺������ͣ�����
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//��ֹ׼�����������������ﵽ���ֵʱ��ֹ

	//ѵ��SVM
	//����һ��SVM���ʵ��
	CvSVM SVM;
	//ѵ��ģ�ͣ�����Ϊ���������ݡ���Ӧ��XX��XX��������ǰ�����ù���
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	//��ʾ�о���
	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
		Mat sampleMat = (Mat_<float>(1, 2) << i, j);
		//predict������Ԥ��ģ�����Ϊ������������ֵ���ͣ����ֵΪture������һ��2�������򷵻��о�����ֵ�����򷵻����ǩ����
		float response = SVM.predict(sampleMat);

		if (response == 1)
			image.at<Vec3b>(j, i) = green;
		else if (response == -1)
			image.at<Vec3b>(j, i) = blue;
	}

	//����ѵ������
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);//��Բ
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 128), 5, Scalar(0, 0, 0), thickness, lineType);

	//��ʾ֧������
	thickness = 2;
	lineType = 8;
	//��ȡ֧�������ĸ���
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		//��ȡ��i��֧������
		const float* v = SVM.get_support_vector(i);
		//֧�������õ��������㣬�û�ɫ���б�ע
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image 

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
}

/************************************************************************/
/* ���Դ��룬���ڻ�ȡ�������ݷ�Ϊ��ȵ������֣�һ��������ѵ����һ������������*/
/************************************************************************/
void spliteTrainTest_forTest( CvMLData &mlData, const CvMat * dataCvMat, int numfeatures)
{
	//cout << dataMat << endl;
	//�����ݷ�Ϊ��ȵ������֣�һ��������ѵ����һ������������
	CvTrainTestSplit spl((float)0.5);
	mlData.set_train_test_split(&spl);
	//��ȡѵ������Ͳ��Ծ���
	const CvMat* traindata_idx = mlData.get_train_sample_idx();
	const CvMat* testdata_idx = mlData.get_test_sample_idx();
	Mat mytraindataidx(traindata_idx);
	Mat mytestdataidx(testdata_idx);
	cout << "mytraindataidx:" << mytraindataidx.size() << endl;
	cout << "mytestdataidx:" << mytestdataidx.size() << endl;
	cout << "mytraindataidx:" << mytraindataidx << endl;
	cout << "mytestdataidx:" << mytestdataidx << endl;

	Mat all_Data(dataCvMat); // for containing the whole matrix offered to us by the .data file.
	Mat all_responses = mlData.get_responses(); //��ǩ for containing all the responses.
	cout << "all_responses:" << all_responses.size() << endl;
	cout << "all_responses:" << all_responses << endl;


	//Ӧ���� rows�ɣ���������
	Mat traindata(mytraindataidx.cols, numfeatures, CV_32F); //for containing all the training data.
	Mat trainresponse(mytraindataidx.cols, 1, CV_32S); // for containing all the outputs we are provided for the training data.
	Mat testdata(mytestdataidx.cols, numfeatures, CV_32F); //for containing all the testing data.
	Mat testresponse(mytestdataidx.cols, 1, CV_32S); //for containing all the outputs of the test data. This will be used for evaluation purpose.

	//Fill in the traindata, testdata,trainresponse and testresponse Mats which were defined above.
	// 	for (int i = 0; i < mytraindataidx.cols; i++)
	// 	{
	// 		trainresponse.at<int>(i) = all_responses.at<float>(mytraindataidx.at<int>(i));
	// 		for (int j = 0; j <= numfeatures; j++)
	// 		{
	// 			traindata.at<float>(i, j) = all_Data.at<float>(mytraindataidx.at<int>(i), j);
	// 		}
	// 	}

	// 	for (int i = 0; i < mytestdataidx.cols; i++)
	// 	{
	// 		testresponse.at<int>(i) = all_responses.at<float>(mytestdataidx.at<int>(i));
	// 		for (int j = 0; j <= numfeatures; j++)
	// 		{
	// 			testdata.at<float>(i, j) = all_Data.at<float>(mytestdataidx.at<int>(i), j);
	// 		}
	// 
}


/************************************************************************/
/* ʹ��SVMѵ�����ݣ�������Ԥ��*/
/************************************************************************/
void svmTrainAndPredict(Mat trainingDataMat, Mat labelsMat)
{
	//����֧���������Ĳ���
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;//SVM���ͣ�ʹ��C֧��������
	params.kernel_type = CvSVM::LINEAR;//�˺������ͣ�����
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//��ֹ׼�����������������ﵽ���ֵʱ��ֹ

	//ѵ��SVM
	CvSVM SVM;//����һ��SVM���ʵ��
	cout << "training........." << endl;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);//ѵ��ģ�ͣ�����Ϊ���������ݡ���Ӧ��XX��XX��������ǰ�����ù���

	//��ȡ��������
	int f1 = 10, f2 = 300, f3 = 5111, f4 = 35.4, f5 = 6;
	Mat sampleMat = (Mat_<float>(1, 5) << f1, f2, f3, f4, f5);

	//Ԥ�⣺predict������Ԥ��ģ�����Ϊ������������ֵ���ͣ����ֵΪture������һ��2�������򷵻��о�����ֵ�����򷵻����ǩ����
	float response = SVM.predict(sampleMat);
	cout << "result is:" << response << endl;

	//��ȡ֧�������ĸ���
	int c = SVM.get_support_vector_count();
	//cout << c << endl;
	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);//��ȡ��i��֧������
		cout << v[i] << endl;
	}
	waitKey();
}


int main()
{
	//csv �ļ�����
	CvMLData mlData;
	mlData.read_csv("������¼.csv");
	const CvMat *dataCvMat = mlData.get_values();
	Mat allDataMat = dataCvMat;  //��ȡ�������ݷ������
	int numFeatures = allDataMat.cols - 1;//��������
	//cout << allDataMat.size() << endl;
	mlData.set_response_idx(numFeatures); //ָ����ǩΪ��һ�У����һ�У�
	Mat trainResponses = mlData.get_responses(); //��ȡ��ǩ����
	Mat trainData(allDataMat.rows, allDataMat.cols - 1, CV_32F);  //�����С
	for (int i = 0; i < allDataMat.cols - 1; i++)//��ȡѵ���õ���������
	{
		for (int j = 0; j < allDataMat.rows; j++)
		{
			trainData.at<float>(j, i) = allDataMat.at<float>(j, i);
		}
	}
	//ѵ��������������ǩ
	Mat trainingDataMat=trainData, labelsMat=trainResponses;
// 	cout << trainingDataMat << endl;
// 	cout << labelsMat << endl;

	//ʹ��SVMѵ�����ݣ�������Ԥ��
	//svmTrainAndPredict(trainingDataMat, labelsMat);

	//
	CvBoost cb;
	cb.train(trainData, CV_ROW_SAMPLE, labelsMat);
	//��ȡ��������
	int f1 = 10, f2 = 300, f3 = 5111, f4 = 35.4, f5 = 6;
	Mat sampleMat = (Mat_<float>(1, 5) << f1, f2, f3, f4, f5);
	//Ԥ�⣺predict������Ԥ��ģ�����Ϊ������������ֵ���ͣ����ֵΪture������һ��2�������򷵻��о�����ֵ�����򷵻����ǩ����
	float response = cb.predict(sampleMat);
	cout << "result is:" << response << endl;

	return 0;

}