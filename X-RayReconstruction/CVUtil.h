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
}