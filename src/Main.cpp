#include <iostream>
#include  <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main() {
	/*char   buffer[FILENAME_MAX];
	cout << _getcwd(buffer, FILENAME_MAX) << endl;*/

	cout << "Hello OpenCV" << endl;

	Mat img = imread("D:\\Develop\\workspace\\HiOpenCV\\res\\plate_num.jpg");
	namedWindow("Pic");
	imshow("Pic", img);

	//convert image to gray
	Mat img_gray;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	blur(img_gray, img_gray, Size(5, 5));
	img=img_gray;
	imshow("Pic_gray", img);

//	Sobel()

	waitKey(0);
	return 0;
}
