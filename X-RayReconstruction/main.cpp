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

	//---------------------------------------
	//	2. 1次元X線投影像を全周方向で撮影
	//---------------------------------------
	const int div_rotation = 360;						//	360分割して撮影
	Mat projectedImage(820, div_rotation, CV_64FC1);	//	投影像ベクトルの集合　列数が角度
	//	回転角theta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	角度radに変換（全部で180deg）
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
	Mat projectedImage_scaled;
	normalize(projectedImage, projectedImage_scaled, 0, 1, CV_MINMAX);
	imshow("X線投影像", projectedImage_scaled);
	waitKey();
	//	画像の保存
	projectedImage.convertTo(projectedImage_scaled, CV_8UC1);
	imwrite("x-ray_projection.png", projectedImage_scaled);

	//---------------------------------------------------------------------------------------------
	//	3. 投影像からの再構成（フーリエ変換法）
	//	投影像の角度固定1次元(r)フーリエ変換＝元画像の2次元(x,y)フーリエ変換　という数学的事実を利用する．
	//	角度thでの投影像の１次元フーリエ変換結果を，周波数平面uv上の角度thの方向に並べてから
	//	平面uvを２次元逆フーリエ変換すると元画像が復元される．
	//---------------------------------------------------------------------------------------------
	////	3.1 まずはDFTに慣れるために元画像のFFTを行う（自分の学習用）．
	////	DFTに最適なサイズを取得（元画像より大きい）
	//Size optDFTSize(getOptimalDFTSize(src.cols),getOptimalDFTSize(src.rows));
	//Mat optDFTImg;		//	元画像の余りを0で埋めた画像
	//copyMakeBorder(src, optDFTImg, 0, optDFTSize.height - src.rows, 0, optDFTSize.width - src.cols, BORDER_CONSTANT, Scalar::all(0));
	////複素数画像complexImg（実部ch/虚部chの2ch）を生成
	//Mat complexPlanes[] = { Mat_<float>(optDFTImg), Mat::zeros(optDFTSize, CV_32F) };
	//Mat complexImg;		//	ここにDFTの結果が入る
	//merge(complexPlanes, 2, complexImg);
	////	DFT実行（実際にはFFT）
	//dft(complexImg, complexImg, DFT_ROWS);
	////	DFT結果表示用にパワースペクトル画像に変換
	////	複素数のL2ノルムの対数
	//split(complexImg, complexPlanes);
	//Mat magImg;
	//magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	複素数要素のノルムを格納
	//magImg += Scalar::all(1);	//	対数化のためのオフセット
	//log(magImg, magImg);		//	値を対数化
	////	DFT画像は四隅が低周波になって出力されるので，中央を直流成分に見せるために入れ替える
	//magImg = magImg(Rect(0, 0, magImg.cols & -2, magImg.rows & -2));	//	DFT最適サイズは奇数の時もある
	//Point2d center(src.cols / 2.0, src.rows / 2.0);
	//Mat tmp;
	//Mat q0(magImg, Rect(0, 0, center.x, center.y));
	//Mat q1(magImg, Rect(center.x, 0, center.x, center.y));
	//Mat q2(magImg, Rect(0, center.y, center.x, center.y));
	//Mat q3(magImg, Rect(center.x, center.y, center.x, center.y));
	//cvutil::swapMat(q0, q3); cvutil::swapMat(q1, q2);
	//normalize(magImg, magImg, 0, 1, CV_MINMAX);
	//imshow("元画像のFFT結果", magImg);
	////	結果の保存
	//normalize(magImg, magImg, 0, 255, CV_MINMAX);
	//magImg.convertTo(magImg, CV_8UC1);
	//imwrite("FFT_spectrum_from_src.png", magImg);
	////	逆フーリエ変換
	//Mat invDFTImg;
	//Mat invDFTplanes[] = { Mat_<float>(optDFTImg), Mat::zeros(optDFTSize, CV_32F) };
	//dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE + DFT_ROWS);	//	FFT結果をそのまま逆FFTにかける
	//split(invDFTImg, invDFTplanes);		//	逆DFTの結果は実部に出てくる
	//Mat invDFTImg_scaled;
	//normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	//imshow("逆FFT結果", invDFTImg_scaled);
	////	結果の保存
	//normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	//invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	//imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	//waitKey();

	//	3.1 投影像を１次元フーリエ変換する
	Mat optDFTImage_sth = projectedImage.t();		//	行と列を入れ替え（行：r軸正，列：th軸正）
	int dftSize = getOptimalDFTSize(projectedImage.rows);
	copyMakeBorder(optDFTImage_sth, optDFTImage_sth, 0, 0, 0, dftSize - projectedImage.rows, BORDER_CONSTANT, Scalar::all(0));
	Mat optDFTImage_sth_scaled;
	normalize(optDFTImage_sth, optDFTImage_sth_scaled, 0, 1, CV_MINMAX);
	imshow("1次元DFT結果", optDFTImage_sth_scaled);
	waitKey();
	//	１次元フーリエ変換のための複素数画像を生成
	Mat complexDFTImage_sth;		//	DFT結果格納用
	Mat complexDFTPlanes_sth[] = { Mat_<float>(optDFTImage_sth), Mat::zeros(optDFTImage_sth.size(), CV_32F) };
	merge(complexDFTPlanes_sth, 2, complexDFTImage_sth);
	//	行毎に１次元FFTを一気に実行 (r, th)->(s, th)
	dft(complexDFTImage_sth, complexDFTImage_sth, DFT_ROWS);
	//	１次元FFT結果を実部と虚部に分ける
	split(complexDFTImage_sth, complexDFTPlanes_sth);
	//	1次元DFT結果を画像出力
	Mat DFTImage_scaled;
	magnitude(complexDFTPlanes_sth[0], complexDFTPlanes_sth[1], DFTImage_scaled);
	DFTImage_scaled += Scalar::all(1);
	log(DFTImage_scaled, DFTImage_scaled);
	normalize(DFTImage_scaled, DFTImage_scaled, 0, 1, CV_MINMAX);
	imshow("1次元DFT結果", DFTImage_scaled);
	
	//	3.2 １次元フーリエ変換した画像を座標変換して(s,th)->(u,v)平面に変換
	//	DFTに最適なサイズを取得（元画像より大きい）
	Size optimalDFTSize_uv(getOptimalDFTSize(src.cols), getOptimalDFTSize(src.rows));
	Mat complexDFTImage_uv(optimalDFTSize_uv, CV_32FC2, Scalar::all(0));		//	uv座標系
	Point2d center_uv(optimalDFTSize_uv.width / 2, optimalDFTSize_uv.height / 2);
	for (int i = 0; i < optimalDFTSize_uv.height; i++) {
		for (int j = 0; j < optimalDFTSize_uv.width; j++) {
			//	格納後の画素(j, i)に対応する格納前の画像座標をサブピクセル精度で割り出す
			Point2d p_uv(j - center_uv.x, center_uv.y - i);		//	現在の画素のuv座標
			double theta = atan2(p_uv.y, p_uv.x);		//	現在の画素の中心からの角度(rad, [-PI, PI])
			double radius = norm(p_uv);
			Point2d p_sth;		//	対応する(s,th)座標
			p_sth.x = (theta < 0) ? -radius : radius;		//	角度が負のとき:sが負でthは正
			p_sth.y = optDFTImage_sth.rows / CV_PI * abs(theta);	//	180degを投影像の数にマッピング
			//	１次元FFTの結果画像からサンプル
			double re = cvutil::sampleSubPix(complexDFTPlanes_sth[0], p_sth);
			double im = cvutil::sampleSubPix(complexDFTPlanes_sth[1], p_sth);
			complexDFTImage_uv.at<Vec2f>(i, j) = Vec2f(re, im);		//	2次元FFTの(i,j)要素にサンプル結果を代入
		}
	}

	//	3.3 (u,v)平面画像を2次元逆フーリエ変換して元画像を得る
	Mat complexInvDFTImage_xy = complexDFTImage_uv.clone();
	dft(complexDFTImage_uv, complexInvDFTImage_xy, DFT_INVERSE | DFT_SCALE);
	Mat complexInvDFTPlanes_xy[]= { Mat::zeros(optimalDFTSize_uv, CV_32F), Mat::zeros(optimalDFTSize_uv, CV_32F) };
	split(complexInvDFTImage_xy, complexInvDFTPlanes_xy);
	Mat invDFTImage_scaled;
	normalize(complexInvDFTPlanes_xy[0], invDFTImage_scaled, 0, 1, CV_MINMAX);
	imshow("投影像復元結果", invDFTImage_scaled);
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
