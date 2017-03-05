#include <iostream>
#include  <stdio.h>
#include  <direct.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

string cwdPath;
string resultPath;

int main() {
	char buffer[MAX_PATH];
	getcwd(buffer, MAX_PATH);

	cout << "Hello OpenCV" << endl;

	cwdPath = string(buffer);
	resultPath = string(cwdPath).append("\\result");
	cout << cwdPath << endl;
	cout << resultPath << endl;

	Mat input = imread(cwdPath + "\\res\\plate_num.jpg");
	namedWindow("Pic");
	imshow("Pic", input);

	//convert image to gray
	Mat img_gray;
	cvtColor(input, img_gray, CV_BGR2GRAY);
	blur(img_gray, img_gray, Size(5, 5));
	imshow("Pic_gray", img_gray);
	imwrite(resultPath + "\\img_gray.png", img_gray);
	waitKey(0);
	destroyWindow("Pic_gray");

	// Sobel
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);
	imshow("Pic_sobel", img_sobel);
	imwrite(resultPath + "\\img_sobel.png", img_sobel);
	waitKey(0);
	destroyWindow("Pic_sobel");

	// threshold
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255,
			CV_THRESH_OTSU + CV_THRESH_BINARY);
	imshow("Pic_threshold", img_threshold);
	imwrite(resultPath + "\\img_threshold.png", img_threshold);
	waitKey(0);
	destroyWindow("Pic_threshold");

	// close morphological operation
	Mat img_morp;
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_morp, CV_MOP_CLOSE, element);

	imshow("Pic_close_morp", img_morp);
	imwrite(resultPath + "\\img_morp.png", img_morp);
	waitKey(0);
	destroyWindow("Pic_close_morp");

	//Find contours of possibles plates
	vector<vector<Point>> contours;
	findContours(img_morp, contours, // a vector of contours
			CV_RETR_EXTERNAL, // retrieve the external contours
			CV_CHAIN_APPROX_NONE); // all pixels of each contour
	//Start to iterate to each contour found
	vector<vector<Point>>::iterator itc = contours.begin();
	vector<RotatedRect> rects;
	//Remove patch that has no inside limits of aspect ratio and area.
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
	Mat result;
	for (int i = 0; i < rects.size(); i++) {
		//For better rect cropping for each possible box
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
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++) {
			Point seed;
			seed.x = rects[i].center.x + rand() % (int) minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int) minSize - (minSize / 2);
			circle(result, seed, 1, Scalar(0, 255, 255), -1);
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp,
					Scalar(loDiff, loDiff, loDiff),
					Scalar(upDiff, upDiff, upDiff), flags);
		}

		waitKey(0);
		return 0;
	}
}

bool verifySizes(RotatedRect candidate) {
	float error = 0.4;
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area. All other patches are discarded
	int min = 15 * aspect * 15;// minimum area
	int max = 125 * aspect * 125;// maximum area
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
