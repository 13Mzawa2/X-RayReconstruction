#include <opencv2\opencv.hpp>

#pragma region OPENCV3_LIBRARY_LINKER
#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
#pragma endregion

using namespace cv;
using namespace std;

int main(void) 
{
	//	1. 画像の読み込み
	Mat test = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src;
	resize(test, src, Size(640, 480), INTER_CUBIC);
	cout << "loaded image information:\n"
		<< "image size = " << src.size()
		<< endl;
	imshow("test", src);
	waitKey();

	//	2. 1次元X線投影像を全周方向で撮影
	const int div_rotation = 360;						//	360分割して撮影
	Mat projectedImage(820, div_rotation, CV_64FC1);	//	投影像ベクトルの集合　列数が角度
	//	回転角theta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	角度radに変換
		//	投影像位置j
		for (int j = 0; j < projectedImage.rows; j++) {
			double r = j - projectedImage.rows / 2.0;	//	rは動径，jの原点を中央にしたもの
			double pix = 0.0;			//	積算結果格納用
			//	画像中心を原点，画像上向きをY軸正としたときの，積算開始位置
			Point2d origin(r*cos(theta), r*sin(theta));
			//	積算方向単位ベクトル
			Point2d n(-sin(theta), cos(theta));
			//	画素の積算　終了条件は画像範囲外に出たとき
			int count = 0;
			//	正方向に積算
			for (int i = 0; ; i++, count++) {
				Point2d p = origin + i*n;
				Point2d p_src(p.x + src.cols / 2, -p.y + src.rows / 2);
				//	現在の積算位置が画像範囲外に出たら終了
				if (p_src.x < 0 || p_src.x > src.cols-1
					|| p_src.y < 0 || p_src.y > src.rows-1) {
					break;
				}
				pix += (double)src.at<uchar>(p_src);
			}
			//	負方向に積算　ただし中央は重複加算しないように開始位置はずらす
			for (int i = 1; ; i++, count++) {
				Point2d p = origin - i*n;
				Point2d p_src(p.x + src.cols / 2, -p.y + src.rows / 2);
				//	現在の積算位置が画像範囲外に出たら終了
				if (p_src.x < 0 || p_src.x > src.cols-1
					|| p_src.y < 0 || p_src.y > src.rows-1) {
					break;
				}
				pix += (double)src.at<uchar>(p_src);
			}
			projectedImage.at<double>(j, th) = pix / count;		//	積算結果を投影像に投影
		}
		cout << "captureing X-ray... : " << th << " / " << div_rotation << "\r";
	}
	cout << endl << "capture finished!!" << endl;
	//	360deg投影像を表示
	Mat projectedImage8;
	projectedImage.convertTo(projectedImage8, CV_8UC1);
	imshow("X線投影像", projectedImage8);
	waitKey();

	return 0;
}