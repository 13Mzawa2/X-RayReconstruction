#pragma once
#include <opencv2\opencv.hpp>

namespace cvutil
{
	//	グレースケール画像mの要素をサブピクセル座標pから取得する
	//	通常のMat::at()メソッドでは四捨五入された座標から取得されてしまうのでcv::getRectSubPix()を利用する
	//	画像範囲外は境界線と同じとする
	//	サポートする型はCV_32F or CV_8U
	double sampleSubPix(cv::Mat m, cv::Point2d p)
	{
		cv::Mat r;
		cv::getRectSubPix(m, cv::Size(3, 3), p, r);
		if (r.depth() == CV_32F) return (double)r.at<float>(1, 1);
		else return (double)r.at<uchar>(1, 1);
	}
	//	完全にMatの中身を入れ替えるswap関数
	//	通常のswap()はROIの中身までは入れ替えてくれない
	void swapMat(cv::Mat &m1, cv::Mat &m2)
	{
		cv::Mat tmp;
		m1.copyTo(tmp); m2.copyTo(m1); tmp.copyTo(m2);
	}
	void fftShift(cv::Mat srcImg, cv::Mat &dstImg) 
	{
		//	DFT画像は四隅が低周波になって出力されるので，中央を直流成分に見せるために入れ替える
		dstImg = srcImg(cv::Rect(0, 0, srcImg.cols & -2, srcImg.rows & -2));	//	DFT最適サイズは奇数の時もある
		cv::Point2d center(srcImg.cols / 2.0, srcImg.rows / 2.0);
		cv::Mat q0(dstImg, cv::Rect(0, 0, center.x, center.y));
		cv::Mat q1(dstImg, cv::Rect(center.x, 0, center.x, center.y));
		cv::Mat q2(dstImg, cv::Rect(0, center.y, center.x, center.y));
		cv::Mat q3(dstImg, cv::Rect(center.x, center.y, center.x, center.y));
		cvutil::swapMat(q0, q3); cvutil::swapMat(q1, q2);
	}
	void fftShift1D(cv::Mat srcImg, cv::Mat &dstImg)
	{
		//	DFT画像は四隅が低周波になって出力されるので，中央を直流成分に見せるために入れ替える
		dstImg = srcImg(cv::Rect(0, 0, srcImg.cols & -2, srcImg.rows & -2));	//	DFT最適サイズは奇数の時もある
		cv::Point2d center(srcImg.cols / 2.0, srcImg.rows / 2.0);
		cv::Mat q0(dstImg, cv::Rect(0, 0, center.x, center.y*2));
		cv::Mat q1(dstImg, cv::Rect(center.x, 0, center.x, center.y*2));
		cvutil::swapMat(q0, q1);
	}
	void fftMagnitude(cv::Mat srcImg, cv::Mat &dstImg)
	{
		std::vector<cv::Mat> srcPlanes;
		cv::split(srcImg, srcPlanes);
		cv::magnitude(srcPlanes[0], srcPlanes[1], dstImg);
		dstImg += cv::Scalar::all(1);
		cv::log(dstImg, dstImg);
		cv::normalize(dstImg, dstImg, 0, 1, CV_MINMAX);
	}
}