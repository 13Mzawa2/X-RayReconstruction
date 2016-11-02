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

namespace cvutil 
{
	//	グレースケール画像mの要素をサブピクセル座標pから取得する
	//	通常のMat::at()メソッドでは四捨五入された座標から取得されてしまうのでcv::getRectSubPix()を利用する
	//	画像範囲外は境界線と同じとする
	//	サポートする型はCV_32F or CV_8U
	double sampleSubPix(cv::Mat m, cv::Point2d p)
	{
		Mat r;
		cv::getRectSubPix(m, cv::Size(3,3), p, r);
		if (r.depth() == CV_32F) return (double)r.at<float>(1, 1);
		else return (double)r.at<uchar>(1, 1);
	}
}

int main(void) 
{
	//	1. 画像の読み込み
	Mat test = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src, srcf;		//	実際の演算はsrcfの画素値を使う
	resize(test, src, Size(640, 480), INTER_CUBIC);
	src.convertTo(srcf, CV_32F);
	cout << "loaded image information:\n"
		<< "image size = " << src.size()
		<< endl;
	imshow("test", src);
	waitKey();
	//	画像の保存
	imwrite("input.png", src);

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
				//pix += (double)src.at<uchar>(p_src);
				pix += cvutil::sampleSubPix(srcf, p_src);
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
				//pix += (double)src.at<uchar>(p_src);
				pix += cvutil::sampleSubPix(srcf, p_src);
			}
			projectedImage.at<double>(j, th) = pix / count;		//	積算結果を投影直線に投影
		}
		//	処理の描画用
		Mat xraycap(src.size(), CV_8UC3);
		cvtColor(src, xraycap, COLOR_GRAY2BGR);
		Point2d center(src.cols / 2.0, src.rows / 2.0); 
		Point2d rminus(center.x - center.x*cos(theta), center.y + center.y*sin(theta));
		Point2d rplus(center.x + center.x*cos(theta), center.y - center.y*sin(theta));
		line(xraycap, rminus, rplus, Scalar(0, 0, 255), 1, CV_AA);
		imshow("test", xraycap);
		waitKey(1);
		cout << "captureing X-ray... : " << th << " / " << div_rotation << "\r";
	}
	imshow("test", src);
	cout << endl << "capture finished!!" << endl;
	//	360deg投影像を表示
	Mat projectedImage8;
	projectedImage.convertTo(projectedImage8, CV_8UC1);
	imshow("X線投影像", projectedImage8);
	waitKey();
	//	画像の保存
	imwrite("x-ray_projection.png", projectedImage8);


	return 0;
}
