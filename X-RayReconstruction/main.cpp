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
	Mat projectedImage = Mat(div_rotation, 880, CV_64FC1, Scalar::all(0));	//	投影像ベクトルの集合 行数が角度
	cout << "投影像サイズ : " << projectedImage.size() << endl;
	//	回転角theta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	角度radに変換（全部で180deg）
		//	画像を回転
		Mat rotatedImage;
		int offsety = projectedImage.cols - srcd.rows;
		int offsetx = projectedImage.cols - srcd.cols;
		copyMakeBorder(srcd, rotatedImage, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));//	画像中心と回転中心を合わせる
		Mat affine = getRotationMatrix2D(Point2d(rotatedImage.cols / 2, rotatedImage.rows / 2), -theta / CV_PI * 180, 1.0);		//	アフィン変換行列
		warpAffine(rotatedImage, rotatedImage, affine, rotatedImage.size(), 1, 0, Scalar::all(0));
		//	画像下からX線を当てて画像上部で撮影
		Mat reduceRow(Size(rotatedImage.cols,1), CV_64FC1, Scalar::all(0));
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
	//	3. 投影像からの再構成（フーリエ変換法）
	//	投影像の角度固定1次元(r)フーリエ変換＝元画像の2次元(x,y)フーリエ変換　という数学的事実（投影定理）を利用する．
	//	角度thでの投影像の１次元フーリエ変換結果を，周波数平面uv上の角度thの方向に並べてから
	//	平面uvを２次元逆フーリエ変換すると元画像が復元される．
	//---------------------------------------------------------------------------------------------
	//	3.0 まずはDFTに慣れるために元画像のFFTを行う（自分の学習用）．
	//	DFTに最適なサイズを取得（元画像より大きい）
	Size optDFTSize(projectedImage.cols, projectedImage.cols);//(getOptimalDFTSize(projectedImage.cols),getOptimalDFTSize(projectedImage.cols));
	Mat optDFTImg;		//	元画像の余りを0で埋めた画像
	int offsety = projectedImage.cols - srcd.rows;
	int offsetx = projectedImage.cols - srcd.cols;
	copyMakeBorder(srcd, optDFTImg, offsety / 2, offsety / 2, offsetx / 2, offsetx / 2, BORDER_CONSTANT, Scalar::all(0));
	//複素数画像complexImg（実部ch/虚部chの2ch）を生成
	Mat complexPlanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTImg.size(), CV_64F) };
	Mat complexImg;		//	ここにDFTの結果が入る
	merge(complexPlanes, 2, complexImg);
	//	DFT実行（実際にはFFT）
	dft(complexImg, complexImg);
	cvutil::fftShift(complexImg, complexImg);		//	FFT結果は象限が反転しているので直す
	//	DFT結果表示用にパワースペクトル画像に変換
	//	複素数のL2ノルムの対数
	split(complexImg, complexPlanes);
	Mat magImg;
	magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	複素数要素のノルムを格納
	magImg += Scalar::all(1);	//	対数化のためのオフセット
	log(magImg, magImg);		//	値を対数化
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	imshow("元画像のFFT結果", magImg);
	//	結果の保存
	normalize(magImg, magImg, 0, 255, CV_MINMAX);
	magImg.convertTo(magImg, CV_8UC1);
	imwrite("FFT_spectrum_from_src.png", magImg);
	//	逆フーリエ変換
	Mat invDFTImg = complexImg;
	Mat invDFTplanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTImg.size(), CV_64F) };
	cvutil::fftShift(invDFTImg, complexImg);
	dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE);	//	FFT結果をそのまま逆FFTにかける
	split(invDFTImg, invDFTplanes);		//	逆DFTの結果は実部に出てくる
	Mat invDFTImg_scaled;
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	imshow("元画像の逆FFT結果", invDFTImg_scaled);
	//	結果の保存
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	waitKey();

	//	1. 撮影画像g(r, th)を1次元フーリエ変換してG(s, th)を得る
	double optDFTSize_rth = projectedImage.cols;//getOptimalDFTSize(projectedImage.cols);
	Mat optDFTImage_rth;
	copyMakeBorder(projectedImage, optDFTImage_rth, 0, 0, 0, optDFTSize_rth - projectedImage.cols, BORDER_CONSTANT, Scalar::all(0));
	//	複素数画像を作成
	Mat complexDFTPlanes_rth[] = { Mat_<double>(optDFTImage_rth), Mat::zeros(optDFTImage_rth.size(), CV_64F) };
	Mat complexDFTImage_rth;
	merge(complexDFTPlanes_rth, 2, complexDFTImage_rth);
	//	1次元DFT実行
	dft(complexDFTImage_rth, complexDFTImage_rth, DFT_ROWS);
	cvutil::fftShift1D(complexDFTImage_rth, complexDFTImage_rth);	//	FFT画像反転
	split(complexDFTImage_rth, complexDFTPlanes_rth);
	//	DFT結果表示
	Mat magImage_rth;
	magnitude(complexDFTPlanes_rth[0], complexDFTPlanes_rth[1], magImage_rth);
	magImage_rth += Scalar::all(1);
	log(magImage_rth, magImage_rth);
	normalize(magImage_rth, magImage_rth, 0, 1, CV_MINMAX);
	imshow("X線投影像1次元FFT結果", magImage_rth);
	//	結果の保存
	normalize(magImage_rth, magImage_rth, 0, 255, CV_MINMAX);
	magImage_rth.convertTo(magImage_rth, CV_8UC1);
	imwrite("FFT1D_projection_rth.png", magImage_rth);

	//	2. G(s, th)をF(u,v)空間に並べ替える
	//	u = s * cos(th), v = s * sin(th)
	Size optDFTSize_uv(optDFTSize_rth, optDFTSize_rth);
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
			double theta = atan2(p_uv.y, p_uv.x);		//	[-PI, PI]
			//double radius = p_uv.x * cos(theta) + p_uv.y * sin(theta);
			double radius = norm(p_uv);
			Point2d p_sth;					//	p_uvに対応するG(s, th)でのサンプル点
			//	原点からの距離が投影像のr軸よりも大きい場合は0とする
			if (radius > optDFTSize_rth / 2.0) {
				complexDFTPlanes_uv[0].at<double>(i, j) = 0.0;	//	実部
				complexDFTPlanes_uv[1].at<double>(i, j) = 0.0;	//	虚部
				continue;
			}
			//	p_uvを変数変換してp_sthに代入
			p_sth.x = (theta > 0) ? center_uv.x + radius : center_uv.x - radius;
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
	//	結果の保存
	normalize(magImage_uv, magImage_uv, 0, 255, CV_MINMAX);
	magImage_uv.convertTo(magImage_uv, CV_8UC1);
	imwrite("FFT1D_projection_xy.png", magImage_uv);

	//	3. F(u,v)からf(x,y)を復元
	Mat complexInvDFTImage;
	cvutil::fftShift(complexDFTImage_uv, complexDFTImage_uv);
	dft(complexDFTImage_uv, complexInvDFTImage, DFT_INVERSE + DFT_SCALE);
	Mat complexInvDFTPlanes[] = { Mat::zeros(optDFTSize_uv, CV_64F), Mat::zeros(optDFTSize_uv, CV_64F) };
	split(complexInvDFTImage, complexInvDFTPlanes);
	Mat invImage_scaled;
	normalize(complexInvDFTPlanes[0], invImage_scaled, 0, 1, CV_MINMAX);
	imshow("投影像からの復元結果", invImage_scaled);
	//	結果の保存
	normalize(invImage_scaled, invImage_scaled, 0, 255, CV_MINMAX);
	invImage_scaled.convertTo(invImage_scaled, CV_8UC1);
	imwrite("inverseFFT2D_projection_xy.png", invImage_scaled);
	waitKey();

	//---------------------------------------------------------------------------------------------
	//	4. 投影像からの再構成（フィルタ補正逆投影法）
	//	周波数領域での積が空間領域での畳込み演算として記述可能なことを利用する．
	//	フーリエ変換法における投影像の１次元フーリエ変換は，投影像に高域強調フィルタ|s|を畳み込む演算と
	//	数学的に同値なので，フィルタ|s|を畳み込んだ投影像を平面xy上で角度thの波としてthで積分すると
	//	元画像が平面xyに復元される．
	//---------------------------------------------------------------------------------------------
	//	G(s, th)に高域強調フィルタをかけて逆変換
	Mat complexDFTImage_rth_s = complexDFTImage_rth.clone();
	for (int i = 0; i < complexDFTImage_rth_s.cols; i++) {
		double d = abs(i - complexDFTImage_rth_s.cols/2);
		complexDFTImage_rth_s.col(i) *= d;
	}
	Mat complexDFTPlanes_rth_s[]
		= { Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1), Mat::zeros(complexDFTImage_rth_s.size(), CV_64FC1) };
	Mat magImage_rth_s;
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	magnitude(complexDFTPlanes_rth_s[0], complexDFTPlanes_rth_s[1], magImage_rth_s);
	magImage_rth_s += Scalar::all(1);
	log(magImage_rth_s, magImage_rth_s);
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s, 0, 1, CV_MINMAX);
	imshow("X線投影像1次元FFT+高域強調フィルタ", magImage_rth_s);
	waitKey();
	//	逆DFT
	cvutil::fftShift1D(complexDFTImage_rth_s, complexDFTImage_rth_s);
	dft(complexDFTImage_rth_s, complexDFTImage_rth_s, DFT_INVERSE + DFT_SCALE + DFT_ROWS);
	//	DFT結果表示
	split(complexDFTImage_rth_s, complexDFTPlanes_rth_s);
	Mat magImage_rth_s_inv;
	normalize(complexDFTPlanes_rth_s[0], magImage_rth_s_inv, 0, 1, CV_MINMAX);
	imshow("X線投影像1次元FFT+高域強調フィルタ", magImage_rth_s_inv);

	//	積算
	Mat invDFTImage_filter = Mat::zeros(Size(projectedImage.cols,projectedImage.cols), CV_64FC1);
	for (int th = 0; th < div_rotation; th++) {
		double theta = th * CV_PI / div_rotation;		//	[0, PI]
		Point2d center(invDFTImage_filter.cols / 2 - 0.5, invDFTImage_filter.rows / 2 - 0.5);
		for (int i = 0; i < invDFTImage_filter.rows; i++) {
			for (int j = 0; j < invDFTImage_filter.cols; j++) {
				Point2d p(j - center.x, -i + center.y);
				//	r軸の法線方向に同じ値を加える
				double r = p.x*cos(theta) + p.y*sin(theta);
				if (r < -center.x)r = -center.x; else if (r >= center.x) r = center.x;
				invDFTImage_filter.at<double>(i, j) += complexDFTPlanes_rth_s[0].at<double>(th, (int)(r+center.x));
			}
		}
		Mat invDFTImage_filter_scaled;
		normalize(invDFTImage_filter, invDFTImage_filter_scaled, 0, 255, CV_MINMAX);
		invDFTImage_filter_scaled.convertTo(invDFTImage_filter_scaled, CV_8U);
		imshow("高域強調フィルタの結果", invDFTImage_filter_scaled);
		cout << "reconstructing... : " << th << " / " << div_rotation << "\r";
		waitKey(1);
	}
	cout << "reconstruction finished!" << endl;
	normalize(invDFTImage_filter, invDFTImage_filter, 0, 255, CV_MINMAX);
	invDFTImage_filter.convertTo(invDFTImage_filter, CV_8U);
	imwrite("s_filter_reconstruction.png", invDFTImage_filter);
	waitKey();
	
	return 0;
}
