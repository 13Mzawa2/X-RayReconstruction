#pragma once
#include <opencv2\opencv.hpp>

namespace cvutil
{
	//	�O���[�X�P�[���摜m�̗v�f���T�u�s�N�Z�����Wp����擾����
	//	�ʏ��Mat::at()���\�b�h�ł͎l�̌ܓ����ꂽ���W����擾����Ă��܂��̂�cv::getRectSubPix()�𗘗p����
	//	�摜�͈͊O�͋��E���Ɠ����Ƃ���
	//	�T�|�[�g����^��CV_32F or CV_8U
	double sampleSubPix(cv::Mat m, cv::Point2d p)
	{
		cv::Mat r;
		cv::getRectSubPix(m, cv::Size(3, 3), p, r);
		if (r.depth() == CV_32F) return (double)r.at<float>(1, 1);
		else return (double)r.at<uchar>(1, 1);
	}
	//	���S��Mat�̒��g�����ւ���swap�֐�
	//	�ʏ��swap()��ROI�̒��g�܂ł͓���ւ��Ă���Ȃ�
	void swapMat(cv::Mat &m1, cv::Mat &m2)
	{
		cv::Mat tmp;
		m1.copyTo(tmp); m2.copyTo(m1); tmp.copyTo(m2);
	}
	void fftShift(cv::Mat srcImg, cv::Mat &dstImg) 
	{
		//	DFT�摜�͎l��������g�ɂȂ��ďo�͂����̂ŁC�����𒼗������Ɍ����邽�߂ɓ���ւ���
		dstImg = srcImg(cv::Rect(0, 0, srcImg.cols & -2, srcImg.rows & -2));	//	DFT�œK�T�C�Y�͊�̎�������
		cv::Point2d center(srcImg.cols / 2.0, srcImg.rows / 2.0);
		cv::Mat q0(dstImg, cv::Rect(0, 0, center.x, center.y));
		cv::Mat q1(dstImg, cv::Rect(center.x, 0, center.x, center.y));
		cv::Mat q2(dstImg, cv::Rect(0, center.y, center.x, center.y));
		cv::Mat q3(dstImg, cv::Rect(center.x, center.y, center.x, center.y));
		cvutil::swapMat(q0, q3); cvutil::swapMat(q1, q2);
	}
	void fftShift1D(cv::Mat srcImg, cv::Mat &dstImg)
	{
		//	DFT�摜�͎l��������g�ɂȂ��ďo�͂����̂ŁC�����𒼗������Ɍ����邽�߂ɓ���ւ���
		dstImg = srcImg(cv::Rect(0, 0, srcImg.cols & -2, srcImg.rows & -2));	//	DFT�œK�T�C�Y�͊�̎�������
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