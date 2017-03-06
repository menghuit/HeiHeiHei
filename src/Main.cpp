#include <iostream>
#include <stdio.h>

#ifdef linux
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define MAX_SIZE (PATH_MAX+1)
#endif

#ifdef WIN32
#include <io.h>
#include <direct.h>
#define MAX_SIZE (MAX_PATH+1)
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

string cwdPath;
string resultPath;

bool verifySizes(RotatedRect candidate) {
	float error = 0.4;
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 15 * aspect * 15;		// minimum area
	int max = 125 * aspect * 125;		// maximum area
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;
	int area = candidate.size.height * candidate.size.width;
	float r = (float) candidate.size.width / (float) candidate.size.height;
	if (r < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	} else {
		return true;
	}
}

//直方图均衡化
Mat histeq(Mat in) {
	Mat out(in.size(), in.type());
	if (in.channels() == 3) {
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(in, hsv, CV_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, CV_HSV2BGR);
	} else if (in.channels() == 1) {
		equalizeHist(in, out);
	}
	return out;
}

int main() {
	char buffer[MAX_SIZE];
	getcwd(buffer, MAX_SIZE);

	cout << "Hello OpenCV" << endl;

	cwdPath = string(buffer);
	resultPath = string(cwdPath).append("/result");
	cout << cwdPath << endl;

#ifdef linux
	mkdir(resultPath.data(), 0777);
#endif

	Mat input = imread(cwdPath + "/res/plate_num.jpg");
	char res[20];
	namedWindow("pic");
	imshow("pic", input);
//	waitKey(0);
//	destroyWindow("pic");

	Mat img_gray;
	cvtColor(input, img_gray, CV_BGR2GRAY);

	//apply a Gaussian blur of 5 x 5 and remove noise
	blur(img_gray, img_gray, Size(5, 5));

	//Finde vertical edges. Car plates have high density of vertical lines
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);//xorder=1,yorder=0,kernelsize=3

	//apply a threshold filter to obtain a binary image through Otsu's method
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255,
			CV_THRESH_OTSU + CV_THRESH_BINARY);

	//Morphplogic operation close:remove blank spaces and connect all regions that have a high number of edges
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);

	//Find 轮廓 of possibles plates
	vector<vector<Point> > contours;
	findContours(img_threshold, contours, // a vector of contours
			CV_RETR_EXTERNAL, // 提取外部轮廓
			CV_CHAIN_APPROX_NONE); // all pixels of each contours

	//Start to iterate to each contour founded
	vector<vector<Point> >::iterator itc = contours.begin();
	vector<RotatedRect> rects;

	//Remove patch that are no inside limits of aspect ratio and area.
	while (itc != contours.end()) {
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr)) {
			itc = contours.erase(itc);
		} else {
			++itc;
			rects.push_back(mr);
		}
	}

	// Draw blue contours on a white image
	Mat result;
//	input.copyTo(result);
//	drawContours(result,contours,
//	  -1, // draw all contours
//	  Scalar(255,0,0), // in blue
//	  3); // with a thickness of 1
//	imshow("pic_contours", result);

	for (int i = 0; i < rects.size(); i++) {
		//For better rect cropping for each posible box
		//Make floodfill algorithm because the plate has white background
		//And then we can retrieve more clearly the contour box
		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		//get the min size between width and height
		float minSize =
				(rects[i].size.width < rects[i].size.height) ?
						rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5;
		//initialize rand and get 5 points around center for floodfill algorithm
		srand(time(NULL));
		//Initialize floodfill parameters and variables
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE
				+ CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++) {
			Point seed;
			seed.x = rects[i].center.x + rand() % (int) minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int) minSize - (minSize / 2);
			circle(result, seed, 1, Scalar(0, 255, 255), -1);
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp,
					Scalar(loDiff, loDiff, loDiff),
					Scalar(upDiff, upDiff, upDiff), flags);
		}
//		sprintf(res,"%s/mask%d.jpg",resultPath.data(), i);
//		printf("%s\n", res);
//		imwrite(res,mask);

		//Check new floodfill mask match for a correct patch.
		//Get all points detected for get Minimal rotated Rect
		vector<Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itMask != end; ++itMask)
			if (*itMask == 255)
				pointsInterest.push_back(itMask.pos());

		RotatedRect minRect = minAreaRect(pointsInterest);

		if (verifySizes(minRect)) {
			// rotated rectangle drawing
			Point2f rect_points[4];
			minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4],
						Scalar(0, 0, 255), 1, 8);

			//Get rotation matrix
			float r = (float) minRect.size.width / (float) minRect.size.height;
			float angle = minRect.angle;
			if (r < 1)
				angle = 90 + angle;
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

			//Create and rotate image
			Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(),
					CV_INTER_CUBIC);

			//Crop image
			Size rect_size = minRect.size;
			if (r < 1)
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0,
					INTER_CUBIC);
			//Equalize croped image
			Mat grayResult;
			cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			blur(grayResult, grayResult, Size(3, 3));
			grayResult = histeq(grayResult);
			/*  if(1){
			 stringstream ss(stringstream::in | stringstream::out);
			 ss << "haha" << "_" << i << ".jpg";
			 imwrite(ss.str(), grayResult);
			 }*/
			//output.push_back(Plate(grayResult,minRect.boundingRect()));
		}
	}
	//imshow("car_plate",result);
	waitKey(0);
	return 0;

//	//convert image to gray
//	Mat img_gray;
//	cvtColor(input, img_gray, CV_BGR2GRAY);
//	blur(img_gray, img_gray, Size(5, 5));
//	imshow("Pic_gray", img_gray);
//	imwrite(resultPath + "/img_gray.png", img_gray);
//	waitKey(0);
//	destroyWindow("Pic_gray");
//
//	// Sobel
//	Mat img_sobel;
//	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);
//	imshow("Pic_sobel", img_sobel);
//	imwrite(resultPath + "/img_sobel.png", img_sobel);
//	waitKey(0);
//	destroyWindow("Pic_sobel");
//
//	// threshold
//	Mat img_threshold;
//	threshold(img_sobel, img_threshold, 0, 255,
//			CV_THRESH_OTSU + CV_THRESH_BINARY);
//	imshow("Pic_threshold", img_threshold);
//	imwrite(resultPath + "/img_threshold.png", img_threshold);
//	waitKey(0);
//	destroyWindow("Pic_threshold");
//
//	// close morphological operation
//	Mat img_morp;
//	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
//	morphologyEx(img_threshold, img_morp, CV_MOP_CLOSE, element);
//
//	imshow("Pic_close_morp", img_morp);
//	imwrite(resultPath + "/img_morp.png", img_morp);
//	waitKey(0);
//	destroyWindow("Pic_close_morp");
//
//	//Find contours of possibles plates
//	vector<vector<Point> > contours;
//	findContours(img_morp, contours, // a vector of contours
//			CV_RETR_EXTERNAL, // retrieve the external contours
//			CV_CHAIN_APPROX_NONE); // all pixels of each contour
//	//Start to iterate to each contour found
//	vector<vector<Point> >::iterator itc = contours.begin();
//	vector<RotatedRect> rects;
//	//Remove patch that has no inside limits of aspect ratio and area.
//	while (itc != contours.end()) {
//		//Create bounding rect of object
//		RotatedRect mr = minAreaRect(Mat(*itc));
//		if (!verifySizes(mr)) {
//			itc = contours.erase(itc);
//		} else {
//			++itc;
//			rects.push_back(mr);
//		}
//	}
//	Mat result;
//	for (int i = 0; i < rects.size(); i++) {
//		//For better rect cropping for each possible box
//		//Make floodfill algorithm because the plate has white background
//		//And then we can retrieve more clearly the contour box
//		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
//		//get the min size between width and height
//		float minSize =
//				(rects[i].size.width < rects[i].size.height) ?
//						rects[i].size.width : rects[i].size.height;
//		minSize = minSize - minSize * 0.5;
//		//initialize rand and get 5 points around center for floodfill algorithm
//		srand(time(NULL));
//		//Initialize floodfill parameters and variables
//		Mat mask;
//		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
//		mask = Scalar::all(0);
//		int loDiff = 30;
//		int upDiff = 30;
//		int connectivity = 4;
//		int newMaskVal = 255;
//		int NumSeeds = 10;
//		Rect ccomp;
//		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
//		for (int j = 0; j < NumSeeds; j++) {
//			Point seed;
//			seed.x = rects[i].center.x + rand() % (int) minSize - (minSize / 2);
//			seed.y = rects[i].center.y + rand() % (int) minSize - (minSize / 2);
//			circle(result, seed, 1, Scalar(0, 255, 255), -1);
//			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp,
//					Scalar(loDiff, loDiff, loDiff),
//					Scalar(upDiff, upDiff, upDiff), flags);
//		}
//
//		waitKey(0);
//		return 0;
//	}
}

