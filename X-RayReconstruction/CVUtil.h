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
}