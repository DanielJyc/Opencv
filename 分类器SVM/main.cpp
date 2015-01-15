#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace std;
/************************************************************************/
/* 测试代码，官方例程：
说明:http://blog.csdn.net/yang_xian521/article/details/6969904
h和：*/
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

	//设置支持向量机的参数
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;//SVM类型：使用C支持向量机
	params.kernel_type = CvSVM::LINEAR;//核函数类型：线性
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//终止准则函数：当迭代次数达到最大值时终止

	//训练SVM
	//建立一个SVM类的实例
	CvSVM SVM;
	//训练模型，参数为：输入数据、响应、XX、XX、参数（前面设置过）
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	//显示判决域
	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
		Mat sampleMat = (Mat_<float>(1, 2) << i, j);
		//predict是用来预测的，参数为：样本、返回值类型（如果值为ture而且是一个2类问题则返回判决函数值，否则返回类标签）、
		float response = SVM.predict(sampleMat);

		if (response == 1)
			image.at<Vec3b>(j, i) = green;
		else if (response == -1)
			image.at<Vec3b>(j, i) = blue;
	}

	//画出训练数据
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);//画圆
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 128), 5, Scalar(0, 0, 0), thickness, lineType);

	//显示支持向量
	thickness = 2;
	lineType = 8;
	//获取支持向量的个数
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
		//获取第i个支持向量
		const float* v = SVM.get_support_vector(i);
		//支持向量用到的样本点，用灰色进行标注
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image 

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
}

/************************************************************************/
/* 测试代码，用于获取。把数据分为相等的两部分，一部分用来训练，一部分用来测试*/
/************************************************************************/
void spliteTrainTest_forTest( CvMLData &mlData, const CvMat * dataCvMat, int numfeatures)
{
	//cout << dataMat << endl;
	//把数据分为相等的两部分，一部分用来训练，一部分用来测试
	CvTrainTestSplit spl((float)0.5);
	mlData.set_train_test_split(&spl);
	//获取训练矩阵和测试矩阵
	const CvMat* traindata_idx = mlData.get_train_sample_idx();
	const CvMat* testdata_idx = mlData.get_test_sample_idx();
	Mat mytraindataidx(traindata_idx);
	Mat mytestdataidx(testdata_idx);
	cout << "mytraindataidx:" << mytraindataidx.size() << endl;
	cout << "mytestdataidx:" << mytestdataidx.size() << endl;
	cout << "mytraindataidx:" << mytraindataidx << endl;
	cout << "mytestdataidx:" << mytestdataidx << endl;

	Mat all_Data(dataCvMat); // for containing the whole matrix offered to us by the .data file.
	Mat all_responses = mlData.get_responses(); //标签 for containing all the responses.
	cout << "all_responses:" << all_responses.size() << endl;
	cout << "all_responses:" << all_responses << endl;


	//应该是 rows吧？？？？？
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
/* 使用SVM训练数据，并进行预测*/
/************************************************************************/
void svmTrainAndPredict(Mat trainingDataMat, Mat labelsMat)
{
	//设置支持向量机的参数
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;//SVM类型：使用C支持向量机
	params.kernel_type = CvSVM::LINEAR;//核函数类型：线性
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//终止准则函数：当迭代次数达到最大值时终止

	//训练SVM
	CvSVM SVM;//建立一个SVM类的实例
	cout << "training........." << endl;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);//训练模型，参数为：输入数据、响应、XX、XX、参数（前面设置过）

	//获取测试数据
	int f1 = 10, f2 = 300, f3 = 5111, f4 = 35.4, f5 = 6;
	Mat sampleMat = (Mat_<float>(1, 5) << f1, f2, f3, f4, f5);

	//预测：predict是用来预测的，参数为：样本、返回值类型（如果值为ture而且是一个2类问题则返回判决函数值，否则返回类标签）、
	float response = SVM.predict(sampleMat);
	cout << "result is:" << response << endl;

	//获取支持向量的个数
	int c = SVM.get_support_vector_count();
	//cout << c << endl;
	for (int i = 0; i < c; ++i)
	{
		const float* v = SVM.get_support_vector(i);//获取第i个支持向量
		cout << v[i] << endl;
	}
	waitKey();
}


int main()
{
	//csv 文件操作
	CvMLData mlData;
	mlData.read_csv("特征记录.csv");
	const CvMat *dataCvMat = mlData.get_values();
	Mat allDataMat = dataCvMat;  //读取所有数据放入矩阵
	int numFeatures = allDataMat.cols - 1;//特征个数
	//cout << allDataMat.size() << endl;
	mlData.set_response_idx(numFeatures); //指定便签为哪一列（最后一列）
	Mat trainResponses = mlData.get_responses(); //获取标签矩阵
	Mat trainData(allDataMat.rows, allDataMat.cols - 1, CV_32F);  //定义大小
	for (int i = 0; i < allDataMat.cols - 1; i++)//获取训练用的特征矩阵
	{
		for (int j = 0; j < allDataMat.rows; j++)
		{
			trainData.at<float>(j, i) = allDataMat.at<float>(j, i);
		}
	}
	//训练数据特征及标签
	Mat trainingDataMat=trainData, labelsMat=trainResponses;
// 	cout << trainingDataMat << endl;
// 	cout << labelsMat << endl;

	//使用SVM训练数据，并进行预测
	//svmTrainAndPredict(trainingDataMat, labelsMat);

	//
	CvBoost cb;
	cb.train(trainData, CV_ROW_SAMPLE, labelsMat);
	//获取测试数据
	int f1 = 10, f2 = 300, f3 = 5111, f4 = 35.4, f5 = 6;
	Mat sampleMat = (Mat_<float>(1, 5) << f1, f2, f3, f4, f5);
	//预测：predict是用来预测的，参数为：样本、返回值类型（如果值为ture而且是一个2类问题则返回判决函数值，否则返回类标签）、
	float response = cb.predict(sampleMat);
	cout << "result is:" << response << endl;

	return 0;

}