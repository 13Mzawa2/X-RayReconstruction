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
	//	�O���[�X�P�[���摜m�̗v�f���T�u�s�N�Z�����Wp����擾����
	//	�ʏ��Mat::at()���\�b�h�ł͎l�̌ܓ����ꂽ���W����擾����Ă��܂��̂�cv::getRectSubPix()�𗘗p����
	//	�摜�͈͊O�͋��E���Ɠ����Ƃ���
	//	�T�|�[�g����^��CV_32F or CV_8U
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
	//	1. �摜�̓ǂݍ���
	Mat test = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src, srcf;		//	���ۂ̉��Z��srcf�̉�f�l���g��
	resize(test, src, Size(640, 480), INTER_CUBIC);
	src.convertTo(srcf, CV_32F);
	cout << "loaded image information:\n"
		<< "image size = " << src.size()
		<< endl;
	imshow("test", src);
	waitKey();
	//	�摜�̕ۑ�
	imwrite("input.png", src);

	//	2. 1����X�����e����S�������ŎB�e
	const int div_rotation = 360;						//	360�������ĎB�e
	Mat projectedImage(820, div_rotation, CV_64FC1);	//	���e���x�N�g���̏W���@�񐔂��p�x
	//	��]�ptheta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	�p�xrad�ɕϊ�
		//	���e���ʒuj
		for (int j = 0; j < projectedImage.rows; j++) {
			double r = j - projectedImage.rows / 2.0;	//	r�͓��a�Cj�̌��_�𒆉��ɂ�������
			double pix = 0.0;			//	�ώZ���ʊi�[�p
			//	�摜���S�����_�C�摜�������Y�����Ƃ����Ƃ��́C�ώZ�J�n�ʒu
			Point2d origin(r*cos(theta), r*sin(theta));
			//	�ώZ�����P�ʃx�N�g��
			Point2d n(-sin(theta), cos(theta));
			//	��f�̐ώZ�@�I�������͉摜�͈͊O�ɏo���Ƃ�
			int count = 0;
			//	�������ɐώZ
			for (int i = 0; ; i++, count++) {
				Point2d p = origin + i*n;
				Point2d p_src(p.x + src.cols / 2, -p.y + src.rows / 2);
				//	���݂̐ώZ�ʒu���摜�͈͊O�ɏo����I��
				if (p_src.x < 0 || p_src.x > src.cols-1
					|| p_src.y < 0 || p_src.y > src.rows-1) {
					break;
				}
				//pix += (double)src.at<uchar>(p_src);
				pix += cvutil::sampleSubPix(srcf, p_src);
			}
			//	�������ɐώZ�@�����������͏d�����Z���Ȃ��悤�ɊJ�n�ʒu�͂��炷
			for (int i = 1; ; i++, count++) {
				Point2d p = origin - i*n;
				Point2d p_src(p.x + src.cols / 2, -p.y + src.rows / 2);
				//	���݂̐ώZ�ʒu���摜�͈͊O�ɏo����I��
				if (p_src.x < 0 || p_src.x > src.cols-1
					|| p_src.y < 0 || p_src.y > src.rows-1) {
					break;
				}
				//pix += (double)src.at<uchar>(p_src);
				pix += cvutil::sampleSubPix(srcf, p_src);
			}
			projectedImage.at<double>(j, th) = pix / count;		//	�ώZ���ʂ𓊉e�����ɓ��e
		}
		//	�����̕`��p
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
	//	360deg���e����\��
	Mat projectedImage8;
	projectedImage.convertTo(projectedImage8, CV_8UC1);
	imshow("X�����e��", projectedImage8);
	waitKey();
	//	�摜�̕ۑ�
	imwrite("x-ray_projection.png", projectedImage8);


	return 0;
}
