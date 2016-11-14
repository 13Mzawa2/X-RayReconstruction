#include <opencv2\opencv.hpp>
#include "CVUtil.h"
//#pragma region OPENCV3_LIBRARY_LINKER
//#ifdef _DEBUG
//#define CV_EXT "d.lib"
//#else
//#define CV_EXT ".lib"
//#endif
//#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
//#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
//#pragma endregion

using namespace cv;
using namespace std;

const int div_rotation = 360;						//	360分割して撮影
const int projectionRowSize = 820;						//	投影像の幅

int main(void)
{
	//	1. 画像の読み込み
	Mat src_read = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src, srcf, srcd;		//	実際の演算はsrcfの画素値を使う
	resize(src_read, src, Size(640, 480), INTER_CUBIC);
	src.convertTo(srcf, CV_32F); src.convertTo(srcd, CV_64F);
	cout << "loaded image information:\n"
		<< "image size = " << src.size()
		<< endl;
	imshow("test", src);
	waitKey();
	//	画像の保存
	imwrite("input.png", src);

	//---------------------------------------
	//	2. 1次元X線投影像を全周方向で撮影
	//---------------------------------------
	Mat projectedImage = Mat(div_rotation, projectionRowSize, CV_64FC1, Scalar::all(0));	//	投影像ベクトルの集合 行数が角度
	cout << "投影像サイズ : " << projectedImage.size() << endl;
	//	回転角theta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	角度radに変換（全部で180deg）
		//	画像を回転
		Mat rotatedImage;		//	この画像の中央で元画像を回転させる
		int offsety = projectionRowSize - srcd.rows;
		int offsetx = projectionRowSize - srcd.cols;
		copyMakeBorder(srcd, rotatedImage, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));//	画像中心と回転中心を合わせる
		Mat affine = getRotationMatrix2D(Point2d(rotatedImage.cols / 2, rotatedImage.rows / 2), -theta / CV_PI * 180, 1.0);		//	アフィン変換行列
		warpAffine(rotatedImage, rotatedImage, affine, rotatedImage.size(), 1, 0, Scalar::all(0));
		//	画像下からX線を当てて画像上部で撮影
		Mat reduceRow(Size(rotatedImage.cols, 1), CV_64FC1, Scalar::all(0));
		reduce(rotatedImage, reduceRow, 0, REDUCE_SUM);	//　行をそのまま加算
		reduceRow.copyTo(projectedImage.row(th));
		//	撮影中画像を表示
		normalize(rotatedImage, rotatedImage, 0, 1, CV_MINMAX);
		imshow("test", rotatedImage);
		Mat projectedImage_scaled;
		normalize(projectedImage, projectedImage_scaled, 0, 255, CV_MINMAX);
		projectedImage_scaled.convertTo(projectedImage_scaled, CV_8UC1);
		imshow("X線投影像", projectedImage_scaled);
		cout << "Captureing... :" << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	imshow("test", src);
	cout << endl << "Capture finished!!" << endl;
	waitKey();
	//	画像の保存
	Mat projectedImage_scaled;
	normalize(projectedImage, projectedImage_scaled, 0, 255, CV_MINMAX);
	projectedImage_scaled.convertTo(projectedImage_scaled, CV_8U);
	imwrite("x-ray_projection.png", projectedImage_scaled);

	//---------------------------------------------------------------------------------------------
	//	3. 投影像からの再構成（フィルタ補正逆投影法）
	//	周波数領域での積が空間領域での畳込み演算として記述可能なことを利用する．
	//	フーリエ変換法における投影像の１次元フーリエ変換は，投影像に高域強調フィルタ|s|を畳み込む演算と
	//	数学的に同値なので，フィルタ|s|を畳み込んだ投影像を平面xy上で角度thの波としてthで積分すると
	//	元画像が平面xyに復元される．
	//---------------------------------------------------------------------------------------------

	//	3.1. 撮影画像g(r, th)を1次元フーリエ変換してG(s, th)を得る
	Mat optDFTImage_rth = projectedImage.clone();
	//	複素数画像を作成
	Mat complexDFTPlanes_rth[] = { Mat_<double>(optDFTImage_rth), Mat::zeros(optDFTImage_rth.size(), CV_64F) };
	Mat complexDFTImage_rth;
	merge(complexDFTPlanes_rth, 2, complexDFTImage_rth);
	//	1次元DFT実行
	dft(complexDFTImage_rth, complexDFTImage_rth, DFT_ROWS);
	cvutil::fftShift1D(complexDFTImage_rth, complexDFTImage_rth);	//	FFT画像反転
	//	DFT結果表示
	Mat magImage_rth;
	cvutil::fftMagnitude(complexDFTImage_rth, magImage_rth);
	imshow("X線投影像1次元FFT結果", magImage_rth);
	//	結果の保存
	normalize(magImage_rth, magImage_rth, 0, 255, CV_MINMAX);
	magImage_rth.convertTo(magImage_rth, CV_8UC1);
	imwrite("FFT1D_projection_rth.png", magImage_rth);

	//	3.2. G(s, th)に高域強調フィルタをかけて逆変換
	Mat complexDFTImage_rth_s = complexDFTImage_rth.clone();
	for (int i = 0; i < complexDFTImage_rth_s.cols; i++) {
		double d = abs(i - complexDFTImage_rth_s.cols / 2 + 0.5);
		complexDFTImage_rth_s.col(i) *= d;
	}
	Mat complexDFTPlanes_rth_s[]
		= { Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1), Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1) };
	//	結果の表示
	Mat magImage_rth_s;
	cvutil::fftMagnitude(complexDFTImage_rth_s, magImage_rth_s);
	imshow("X線投影像1次元FFT+高域強調フィルタ", magImage_rth_s);
	//	逆DFT
	cvutil::fftShift1D(complexDFTImage_rth_s, complexDFTImage_rth_s);
	dft(complexDFTImage_rth_s, complexDFTImage_rth_s, DFT_INVERSE + DFT_SCALE + DFT_ROWS);
	//	DFT結果表示
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	Mat magImage_rth_s_inv;
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s_inv, 0, 1, CV_MINMAX);
	imshow("X線投影像1次元FFT+高域強調フィルタの逆FFT", magImage_rth_s_inv);
	waitKey();

	//	3.3. 空間周波数を積算
	Mat invProjectedImage_filter = Mat::zeros(Size(projectionRowSize, projectionRowSize), CV_64FC1);
	for (int th = 0; th < div_rotation; th++) {
		double theta = th * CV_PI / div_rotation;		//	[0, PI]
		Point2d center(invProjectedImage_filter.cols / 2 - 0.5, invProjectedImage_filter.rows / 2 - 0.5);
		//	theta方向の空間周波数画像を作成して積算
		for (int i = 0; i < invProjectedImage_filter.rows; i++) {
			for (int j = 0; j < invProjectedImage_filter.cols; j++) {
				Point2d p(j - center.x, -i + center.y);
				//	r軸の法線方向に同じ値を加える
				double r = p.x*cos(theta) + p.y*sin(theta);
				if (r < -center.x)r = -center.x; else if (r >= center.x) r = center.x;
				invProjectedImage_filter.at<double>(i, j) += complexDFTPlanes_rth_s[0].at<double>(th, (int)(r + center.x));
			}
		}
		Mat invProjectedImage_filter_scaled;
		normalize(invProjectedImage_filter, invProjectedImage_filter_scaled, 0, 255, CV_MINMAX);
		invProjectedImage_filter_scaled.convertTo(invProjectedImage_filter_scaled, CV_8U);
		imshow("高域強調フィルタの結果", invProjectedImage_filter_scaled);
		cout << "reconstructing... : " << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	cout << "reconstruction finished!" << endl;
	normalize(invProjectedImage_filter, invProjectedImage_filter, 0, 255, CV_MINMAX);
	invProjectedImage_filter.convertTo(invProjectedImage_filter, CV_8U);
	imwrite("s_filter_reconstruction.png", invProjectedImage_filter);
	waitKey();

	return 0;
}
