#include <opencv2\opencv.hpp>
#include "CVUtil.h"
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
	const int div_rotation = 360;						//	360分割して撮影
	Mat projectedImage = Mat::zeros(div_rotation, 480, CV_64FC1);	//	投影像ベクトルの集合　行数が角度
	//	回転角theta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	角度radに変換（全部で180deg）
		//	画像を回転
		Mat rotatedImage;
		Mat affine = getRotationMatrix2D(Point2d(src.cols / 2, src.rows / 2), -theta / CV_PI * 180, 1.0);		//	アフィン変換行列
		warpAffine(srcd, rotatedImage, affine, src.size());
		//	画像下からX線を当てて画像上部で撮影
		Mat reduceRow, clippedRow;
		reduce(rotatedImage, reduceRow, 0, REDUCE_AVG);	//　行をそのまま平均化
		if (projectedImage.cols < reduceRow.cols) {
			for (int i = 0; i < projectedImage.cols; i++) {
				projectedImage.row(th).at<double>(i)
					= reduceRow.at<double>(i + reduceRow.cols / 2 - projectedImage.cols / 2);
			}
		}
		else {
			for (int i = 0; i < reduceRow.cols; i++) {
				projectedImage.row(th).at<double>(i - reduceRow.cols / 2 + projectedImage.cols / 2)
					= reduceRow.at<double>(i);
			}
		}
		//	撮影中画像を表示
		normalize(rotatedImage, rotatedImage, 0, 1, CV_MINMAX);
		imshow("test", rotatedImage);
		Mat projectedImage_scaled;
		projectedImage.convertTo(projectedImage_scaled, CV_8UC1);
		imshow("X線投影像", projectedImage_scaled);
		waitKey(1);
	}
	imshow("test", src);
	cout << endl << "capture finished!!" << endl;
	//	360deg投影像を表示
	//normalize(projectedImage, projectedImage_scaled, 0, 1, CV_MINMAX);
	waitKey();
	//	画像の保存
	imwrite("x-ray_projection.png", projectedImage);

	//---------------------------------------------------------------------------------------------
	//	3. 投影像からの再構成（フーリエ変換法）
	//	投影像の角度固定1次元(r)フーリエ変換＝元画像の2次元(x,y)フーリエ変換　という数学的事実を利用する．
	//	角度thでの投影像の１次元フーリエ変換結果を，周波数平面uv上の角度thの方向に並べてから
	//	平面uvを２次元逆フーリエ変換すると元画像が復元される．
	//---------------------------------------------------------------------------------------------
	//	3.0 まずはDFTに慣れるために元画像のFFTを行う（自分の学習用）．
	//	DFTに最適なサイズを取得（元画像より大きい）
	Size optDFTSize(getOptimalDFTSize(srcd.cols),getOptimalDFTSize(srcd.rows));
	Mat optDFTImg;		//	元画像の余りを0で埋めた画像
	copyMakeBorder(srcd, optDFTImg, 0, optDFTSize.height - srcd.rows, 0, optDFTSize.width - srcd.cols, BORDER_CONSTANT, Scalar::all(0));
	//複素数画像complexImg（実部ch/虚部chの2ch）を生成
	Mat complexPlanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTSize, CV_64F) };
	Mat complexImg;		//	ここにDFTの結果が入る
	merge(complexPlanes, 2, complexImg);
	//	DFT実行（実際にはFFT）
	dft(complexImg, complexImg);
	cvutil::fftShift(complexImg, complexImg);
	//	DFT結果表示用にパワースペクトル画像に変換
	//	複素数のL2ノルムの対数
	split(complexImg, complexPlanes);
	Mat magImg;
	magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	複素数要素のノルムを格納
	magImg += Scalar::all(1);	//	対数化のためのオフセット
	log(magImg, magImg);		//	値を対数化
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	imshow("元画像のFFT結果", magImg);
	////	結果の保存
	//normalize(magImg, magImg, 0, 255, CV_MINMAX);
	//magImg.convertTo(magImg, CV_8UC1);
	//imwrite("FFT_spectrum_from_src.png", magImg);
	//	逆フーリエ変換
	Mat invDFTImg= complexImg;
	Mat invDFTplanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTSize, CV_64F) };
	cvutil::fftShift(invDFTImg, complexImg);
	dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE);	//	FFT結果をそのまま逆FFTにかける
	split(invDFTImg, invDFTplanes);		//	逆DFTの結果は実部に出てくる
	Mat invDFTImg_scaled;
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	imshow("元画像の逆FFT結果", invDFTImg_scaled);
	////	結果の保存
	//normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	//invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	//imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	waitKey();

	//	1. 撮影画像g(r, th)を1次元フーリエ変換してG(s, th)を得る
	double optDFTSize_rth = getOptimalDFTSize(projectedImage.cols);
	Mat optDFTImage_rth;
	copyMakeBorder(projectedImage, optDFTImage_rth, 0, 0, 0, optDFTSize_rth - projectedImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//	複素数画像を作成
	Mat complexDFTPlanes_rth[] = { Mat_<double>(optDFTImage_rth), Mat::zeros(optDFTImage_rth.size(), CV_64F) };
	Mat complexDFTImage_rth;
	merge(complexDFTPlanes_rth, 2, complexDFTImage_rth);
	//	1次元DFT実行
	dft(complexDFTImage_rth, complexDFTImage_rth, DFT_ROWS);
	cvutil::fftShift1D(complexDFTImage_rth, complexDFTImage_rth);	//	FFT画像反転
	//	DFT結果表示
	split(complexDFTImage_rth, complexDFTPlanes_rth);
	Mat magImage_rth;
	magnitude(complexDFTPlanes_rth[0], complexDFTPlanes_rth[1], magImage_rth);
	magImage_rth += Scalar::all(1);
	log(magImage_rth, magImage_rth);
	normalize(magImage_rth, magImage_rth, 0, 1, CV_MINMAX);
	imshow("X線投影像1次元FFT結果", magImage_rth);

	//	2. G(s, th)をF(u,v)空間に並べ替える
	//	u = s * cos(th), v = s * sin(th)
	Size optDFTSize_uv(getOptimalDFTSize(src.cols), getOptimalDFTSize(src.rows));
	Mat complexDFTPlanes_uv[] 
		= { Mat::zeros(optDFTSize_uv, CV_64FC1), Mat::zeros(optDFTSize_uv, CV_64FC1) };
	//	サブピクセルサンプリングのためにfloat型に変換
	Mat complexDFTPlane_sth_re, complexDFTPlane_sth_im;
	complexDFTPlanes_rth[0].convertTo(complexDFTPlane_sth_re, CV_32F);
	complexDFTPlanes_rth[1].convertTo(complexDFTPlane_sth_im, CV_32F);
	for (int i = 0; i < optDFTSize_uv.height; i++) {
		for (int j = 0; j < optDFTSize_uv.width; j++) {
			Point2d center_uv(optDFTSize_uv.width / 2, optDFTSize_uv.height / 2);
			Point2d p_uv(j - center_uv.x, - i + center_uv.y);	//	現在のスコープ座標(u,v)
			double radius = norm(p_uv);
			Point2d p_sth;					//	p_uvに対応するG(s, th)でのサンプル点
			//	原点からの距離が投影像のr軸よりも大きい場合は0とする
			if (radius > optDFTSize_rth / 2.0) {
				complexDFTPlanes_uv[0].at<double>(i, j) = 0.0;	//	実部
				complexDFTPlanes_uv[1].at<double>(i, j) = 0.0;	//	虚部
				continue;
			}
			//	p_uvを変数変換してp_sthに代入
			double theta = atan2(p_uv.y, p_uv.x);		//	[-PI, PI]
			p_sth.x = (theta > 0) ? optDFTSize_rth/2.0 + radius : optDFTSize_rth/2.0 - radius;
			p_sth.y = (theta > 0) ? theta*div_rotation / CV_PI : (theta + CV_PI) * div_rotation / CV_PI;

			complexDFTPlanes_uv[0].at<double>(i, j) = cvutil::sampleSubPix(complexDFTPlane_sth_re, p_sth);
			complexDFTPlanes_uv[1].at<double>(i, j) = cvutil::sampleSubPix(complexDFTPlane_sth_im, p_sth);
		}
	}
	Mat complexDFTImage_uv;
	merge(complexDFTPlanes_uv, 2, complexDFTImage_uv);
	//	DFT結果表示
	split(complexDFTImage_uv, complexDFTPlanes_uv);
	Mat magImage_uv;
	magnitude(complexDFTPlanes_uv[0], complexDFTPlanes_uv[1], magImage_uv);
	magImage_uv += Scalar::all(1);
	log(magImage_uv, magImage_uv);
	normalize(magImage_uv, magImage_uv, 0, 1, CV_MINMAX);
	imshow("X線投影像2次元FFT復元結果", magImage_uv);

	//	3. F(u,v)からf(x,y)を復元
	cvutil::fftShift(complexDFTImage_uv, complexDFTImage_uv);
	Mat complexInvDFTImage;
	dft(complexDFTImage_uv, complexInvDFTImage, DFT_INVERSE + DFT_SCALE);
	Mat complexInvDFTPlanes[] = { Mat::zeros(optDFTSize_uv, CV_64F), Mat::zeros(optDFTSize_uv, CV_64F) };
	split(complexInvDFTImage, complexInvDFTPlanes);
	Mat invImage_scaled;
	normalize(complexInvDFTPlanes[0], invImage_scaled, 0, 1, CV_MINMAX);
	imshow("投影像からの復元結果", invImage_scaled);
	waitKey();

	//---------------------------------------------------------------------------------------------
	//	4. 投影像からの再構成（フィルタ補正逆投影法）
	//	周波数領域での積が空間領域での畳込み演算として記述可能なことを利用する．
	//	フーリエ変換法における投影像の１次元フーリエ変換は，投影像に高域強調フィルタ|s|を畳み込む演算と
	//	数学的に同値なので，フィルタ|s|を畳み込んだ投影像を平面xyの角度th上に並べてやると
	//	元画像が平面xyに復元される．
	//---------------------------------------------------------------------------------------------
	
	
	return 0;
}
