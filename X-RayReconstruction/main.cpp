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

	//---------------------------------------
	//	2. 1����X�����e����S�������ŎB�e
	//---------------------------------------
	const int div_rotation = 360;						//	360�������ĎB�e
	Mat projectedImage(820, div_rotation, CV_64FC1);	//	���e���x�N�g���̏W���@�񐔂��p�x
	//	��]�ptheta
	for (int th = 0; th < div_rotation; th++) {
		double theta = CV_PI / div_rotation * th;		//	�p�xrad�ɕϊ��i�S����180deg�j
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
	Mat projectedImage_scaled;
	normalize(projectedImage, projectedImage_scaled, 0, 1, CV_MINMAX);
	imshow("X�����e��", projectedImage_scaled);
	waitKey();
	//	�摜�̕ۑ�
	projectedImage.convertTo(projectedImage_scaled, CV_8UC1);
	imwrite("x-ray_projection.png", projectedImage_scaled);

	//---------------------------------------------------------------------------------------------
	//	3. ���e������̍č\���i�t�[���G�ϊ��@�j
	//	���e���̊p�x�Œ�1����(r)�t�[���G�ϊ������摜��2����(x,y)�t�[���G�ϊ��@�Ƃ������w�I�����𗘗p����D
	//	�p�xth�ł̓��e���̂P�����t�[���G�ϊ����ʂ��C���g������uv��̊p�xth�̕����ɕ��ׂĂ���
	//	����uv���Q�����t�t�[���G�ϊ�����ƌ��摜�����������D
	//---------------------------------------------------------------------------------------------
	//	3.1 �܂���DFT�Ɋ���邽�߂Ɍ��摜��FFT���s���i�����̊w�K�p�j�D
	//	DFT�ɍœK�ȃT�C�Y���擾�i���摜���傫���j
	Size optDFTSize(getOptimalDFTSize(src.cols),getOptimalDFTSize(src.rows));
	Mat optDFTImg;		//	���摜�̗]���0�Ŗ��߂��摜
	copyMakeBorder(src, optDFTImg, 0, optDFTSize.height - src.rows, 0, optDFTSize.width - src.cols, BORDER_CONSTANT, Scalar::all(0));
	//���f���摜complexImg�i����ch/����ch��2ch�j�𐶐�
	Mat complexPlanes[] = { Mat_<float>(optDFTImg), Mat::zeros(optDFTSize, CV_32F) };
	Mat complexImg;		//	������DFT�̌��ʂ�����
	merge(complexPlanes, 2, complexImg);
	//	DFT���s�i���ۂɂ�FFT�j
	dft(complexImg, complexImg, DFT_COMPLEX_OUTPUT);
	//	DFT���ʕ\���p�Ƀp���[�X�y�N�g���摜�ɕϊ�
	//	���f����L2�m�����̑ΐ�
	split(complexImg, complexPlanes);
	Mat magImg;
	magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	���f���v�f�̃m�������i�[
	magImg += Scalar::all(1);	//	�ΐ����̂��߂̃I�t�Z�b�g
	log(magImg, magImg);		//	�l��ΐ���
	//	DFT�摜�͎l��������g�ɂȂ��ďo�͂����̂ŁC�����𒼗������Ɍ����邽�߂ɓ���ւ���
	magImg = magImg(Rect(0, 0, magImg.cols & -2, magImg.rows & -2));	//	DFT�œK�T�C�Y�͊�̎�������
	Point2d center(src.cols / 2.0, src.rows / 2.0);
	Mat tmp;
	Mat q0(magImg, Rect(0, 0, center.x, center.y));
	Mat q1(magImg, Rect(center.x, 0, center.x, center.y));
	Mat q2(magImg, Rect(0, center.y, center.x, center.y));
	Mat q3(magImg, Rect(center.x, center.y, center.x, center.y));
	cvutil::swapMat(q0, q3); cvutil::swapMat(q1, q2);
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	imshow("���摜��FFT����", magImg);
	//	���ʂ̕ۑ�
	normalize(magImg, magImg, 0, 255, CV_MINMAX);
	magImg.convertTo(magImg, CV_8UC1);
	imwrite("FFT_spectrum_from_src.png", magImg);
	//	�t�t�[���G�ϊ�
	Mat invDFTImg;
	Mat invDFTplanes[] = { Mat_<float>(optDFTImg), Mat::zeros(optDFTSize, CV_32F) };
	dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE);	//	FFT���ʂ����̂܂܋tFFT�ɂ�����
	split(invDFTImg, invDFTplanes);		//	�tDFT�̌��ʂ͎����ɏo�Ă���
	Mat invDFTImg_scaled;
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	imshow("�tFFT����", invDFTImg_scaled);
	//	���ʂ̕ۑ�
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	waitKey();

	////	3.2 ���e�����P�����t�[���G�ϊ�����
	//int dftSize = getOptimalDFTSize(projectedImage.rows);
	//Mat optDFTImage = projectedImage.t();		//	�s�Ɨ�����ւ��i�s�Fr�����C��Fth�����j
	//copyMakeBorder(optDFTImage, optDFTImage, 0, 0, 0, dftSize - projectedImage.rows, BORDER_CONSTANT, Scalar::all(0));
	////	�P�����t�[���G�ϊ��̂��߂̕��f���摜�𐶐�
	//Mat DFTPlane;
	//Mat complexDFTPlanes[] = { Mat_<float>(optDFTImage), Mat::zeros(optDFTImage.size(), CV_32F) };
	//merge(complexDFTPlanes, 2, DFTPlane);
	////	�s���ɂP����FFT����C�Ɏ��s
	//dft(DFTPlane, DFTPlane, DFT_COMPLEX_OUTPUT | DFT_ROWS, projectedImage.rows);
	////	�e�s��DFT�摜�Ɋp�x���Ɋi�[
	//Mat invDFTImage;
	//Mat complexInvDFTImage(optDFTSize, CV_32FC2, Scalar::all(0));
	//for (int th = 0; th < div_rotation; th++) {
	//	double theta = CV_PI / div_rotation * th;		//	�p�xrad�ɕϊ��i�S����180deg�j
	//	Mat projectionPlane = projectedImage.col(th);	//	�p�xth�ł̓��e�������o��
	//	
	//	
	//}

	//---------------------------------------------------------------------------------------------
	//	4. ���e������̍č\���i�t�B���^�␳�t���e�@�j
	//	���g���̈�ł̐ς���ԗ̈�ł̏􍞂݉��Z�Ƃ��ċL�q�\�Ȃ��Ƃ𗘗p����D
	//	�t�[���G�ϊ��@�ɂ����铊�e���̂P�����t�[���G�ϊ��́C���e���ɍ��拭���t�B���^|s|����ݍ��މ��Z��
	//	���w�I�ɓ��l�Ȃ̂ŁC�t�B���^|s|����ݍ��񂾓��e���𕽖�xy�̊p�xth��ɕ��ׂĂ���
	//	���摜������xy�ɕ��������D
	//---------------------------------------------------------------------------------------------
	return 0;
}
