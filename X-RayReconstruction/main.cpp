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
	Mat src_read = imread("doge.jpg", IMREAD_GRAYSCALE);
	Mat src, srcf, srcd;		//	���ۂ̉��Z��srcf�̉�f�l���g��
	resize(src_read, src, Size(640, 480), INTER_CUBIC);
	src.convertTo(srcf, CV_32F); src.convertTo(srcd, CV_64F);
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
	Mat projectedImage(460, div_rotation, CV_64FC1);	//	���e���x�N�g���̏W���@�񐔂��p�x
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
	//normalize(projectedImage, projectedImage_scaled, 0, 1, CV_MINMAX);
	projectedImage.convertTo(projectedImage_scaled, CV_8UC1);
	imshow("X�����e��", projectedImage_scaled);
	waitKey();
	//	�摜�̕ۑ�
	imwrite("x-ray_projection.png", projectedImage_scaled);

	//---------------------------------------------------------------------------------------------
	//	3. ���e������̍č\���i�t�[���G�ϊ��@�j
	//	���e���̊p�x�Œ�1����(r)�t�[���G�ϊ������摜��2����(x,y)�t�[���G�ϊ��@�Ƃ������w�I�����𗘗p����D
	//	�p�xth�ł̓��e���̂P�����t�[���G�ϊ����ʂ��C���g������uv��̊p�xth�̕����ɕ��ׂĂ���
	//	����uv���Q�����t�t�[���G�ϊ�����ƌ��摜�����������D
	//---------------------------------------------------------------------------------------------
	//	3.0 �܂���DFT�Ɋ���邽�߂Ɍ��摜��FFT���s���i�����̊w�K�p�j�D
	//	DFT�ɍœK�ȃT�C�Y���擾�i���摜���傫���j
	Size optDFTSize(getOptimalDFTSize(srcd.cols),getOptimalDFTSize(srcd.rows));
	Mat optDFTImg;		//	���摜�̗]���0�Ŗ��߂��摜
	copyMakeBorder(srcd, optDFTImg, 0, optDFTSize.height - srcd.rows, 0, optDFTSize.width - srcd.cols, BORDER_CONSTANT, Scalar::all(0));
	//���f���摜complexImg�i����ch/����ch��2ch�j�𐶐�
	Mat complexPlanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTSize, CV_64F) };
	Mat complexImg;		//	������DFT�̌��ʂ�����
	merge(complexPlanes, 2, complexImg);
	//	DFT���s�i���ۂɂ�FFT�j
	dft(complexImg, complexImg);
	cvutil::fftShift(complexImg, complexImg);
	//	DFT���ʕ\���p�Ƀp���[�X�y�N�g���摜�ɕϊ�
	//	���f����L2�m�����̑ΐ�
	split(complexImg, complexPlanes);
	Mat magImg;
	magnitude(complexPlanes[0], complexPlanes[1], magImg);		//	���f���v�f�̃m�������i�[
	magImg += Scalar::all(1);	//	�ΐ����̂��߂̃I�t�Z�b�g
	log(magImg, magImg);		//	�l��ΐ���
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	imshow("���摜��FFT����", magImg);
	////	���ʂ̕ۑ�
	//normalize(magImg, magImg, 0, 255, CV_MINMAX);
	//magImg.convertTo(magImg, CV_8UC1);
	//imwrite("FFT_spectrum_from_src.png", magImg);
	//	�t�t�[���G�ϊ�
	Mat invDFTImg= complexImg;
	Mat invDFTplanes[] = { Mat_<double>(optDFTImg), Mat::zeros(optDFTSize, CV_64F) };
	cvutil::fftShift(invDFTImg, complexImg);
	dft(complexImg, invDFTImg, DFT_INVERSE + DFT_SCALE);	//	FFT���ʂ����̂܂܋tFFT�ɂ�����
	split(invDFTImg, invDFTplanes);		//	�tDFT�̌��ʂ͎����ɏo�Ă���
	Mat invDFTImg_scaled;
	normalize(invDFTplanes[0], invDFTImg_scaled, 0, 1, CV_MINMAX);
	imshow("���摜�̋tFFT����", invDFTImg_scaled);
	////	���ʂ̕ۑ�
	//normalize(invDFTplanes[0], invDFTImg_scaled, 0, 255, CV_MINMAX);
	//invDFTImg_scaled.convertTo(invDFTImg_scaled, CV_8UC1);
	//imwrite("inverseFFT_from_src.png", invDFTImg_scaled);
	//waitKey();

	//	3.1 ���e�����P�����t�[���G�ϊ�����
	Mat optDFTImage_sth = projectedImage.t();		//	�s�Ɨ�����ւ��i�s�Fr�����C��Fth�����j
	int dftSize = getOptimalDFTSize(optDFTImage_sth.cols);
	copyMakeBorder(optDFTImage_sth, optDFTImage_sth, 0, 0, 0, dftSize - optDFTImage_sth.cols, BORDER_CONSTANT, Scalar::all(0));
	//	�P�����t�[���G�ϊ��̂��߂̕��f���摜�𐶐�
	Mat complexDFTImage_sth;		//	DFT���ʊi�[�p
	Mat complexDFTPlanes_sth[] = { Mat_<double>(optDFTImage_sth), Mat::zeros(optDFTImage_sth.size(), CV_64FC1) };
	merge(complexDFTPlanes_sth, 2, complexDFTImage_sth);
	//	�摜�\��
	cout << "projection image size = " << optDFTImage_sth.size() << endl;
	Mat optDFTImage_sth_scaled;
	normalize(complexDFTPlanes_sth[0], optDFTImage_sth_scaled, 0, 255, CV_MINMAX);
	optDFTImage_sth_scaled.convertTo(optDFTImage_sth_scaled, CV_8UC1);
	imshow("1����DFT����", optDFTImage_sth_scaled);
	waitKey(); 
	//	�s���ɂP����FFT����C�Ɏ��s (r, th)->(s, th)
	dft(complexDFTImage_sth, complexDFTImage_sth, DFT_ROWS, projectedImage.rows);
	cvutil::fftShift1D(complexDFTImage_sth, complexDFTImage_sth);
	//	�P����FFT���ʂ������Ƌ����ɕ�����
	split(complexDFTImage_sth, complexDFTPlanes_sth);

	//	���e�����̂������ł��邩�m�F
	Mat test, test_fft = complexDFTImage_sth.clone();
	Mat testplane[] = { Mat::zeros(optDFTImage_sth.size(), CV_64F),Mat::zeros(optDFTImage_sth.size(), CV_64F) };
	////	�����g�����J�b�g�}�X�N�摜
	//Mat mask = Mat::zeros(test_fft.size(), CV_64FC2);
	//rectangle(mask, Point(50, 0), Point(mask.cols - 50, mask.rows), Scalar::all(1.0), -1);
	//test_fft = test_fft.mul(mask);
	split(test_fft, testplane);
	//	1����DFT���ʂ��摜�o��
	Mat DFTImage_scaled;
	magnitude(testplane[0], testplane[1], DFTImage_scaled);
	DFTImage_scaled += Scalar::all(1);
	log(DFTImage_scaled, DFTImage_scaled);
	normalize(DFTImage_scaled, DFTImage_scaled, 0, 255, CV_MINMAX);
	DFTImage_scaled.convertTo(DFTImage_scaled, CV_8UC1);
	imshow("1����DFT����", DFTImage_scaled);

	//	���拭���t�B���^
	for (int i = 0; i < complexDFTImage_sth.cols; i++) {
		complexDFTImage_sth.col(i) *= abs(i - complexDFTImage_sth.cols / 2);
	}
	cvutil::fftShift1D(test_fft, test_fft);
	dft(test_fft, test, DFT_INVERSE + DFT_SCALE + DFT_ROWS);
	split(test, testplane);
	Mat test_scaled;
	normalize(testplane[0], test_scaled, 0, 1, CV_MINMAX);
	imshow("���e���̋tFFT����", test_scaled);

	//	3.2 �P�����t�[���G�ϊ������摜�����W�ϊ�����(s,th)->(u,v)���ʂɕϊ�
	//	DFT�ɍœK�ȃT�C�Y���擾�i���摜���傫���j
	Size optimalDFTSize_uv(getOptimalDFTSize(src.cols), getOptimalDFTSize(src.rows));
	Mat complexDFTImage_uv(optimalDFTSize_uv, CV_64FC2, Scalar::all(0));		//	uv���W�n
	Point2d center_uv(optimalDFTSize_uv.width / 2, optimalDFTSize_uv.height / 2);
	Mat complexDFTPlanes_sth_re, complexDFTPlanes_sth_im;
	complexDFTPlanes_sth[0].convertTo(complexDFTPlanes_sth_re, CV_32F);
	complexDFTPlanes_sth[1].convertTo(complexDFTPlanes_sth_im, CV_32F);
	for (int i = 0; i < optimalDFTSize_uv.height; i++) {
		for (int j = 0; j < optimalDFTSize_uv.width; j++) {
			//	�i�[��̉�f(j, i)�ɑΉ�����i�[�O�̉摜���W���T�u�s�N�Z�����x�Ŋ���o��
			Point2d p_uv(j - center_uv.x, center_uv.y - i);		//	���݂̉�f��uv���W
			double theta = atan2(p_uv.y, p_uv.x);		//	���݂̉�f�̒��S����̊p�x(rad, [-PI, PI])
			double radius = norm(p_uv);
			double length = dftSize/2.0;			//	radius�̍ő咷��
			if (radius > length) {
				complexDFTImage_uv.at<Vec2d>(i, j) = Vec2d(0, 0);
				continue;
			}
			Point2d p_sth;		//	�Ή�����(s,th)���W
			p_sth.x = (theta > 0) ?
				radius + dftSize/2
				: -radius + dftSize / 2;		//	�p�x�����̂Ƃ�:s������th�͐�
			p_sth.y = (theta > 0) ?
				optDFTImage_sth.rows / CV_PI * theta
				: optDFTImage_sth.rows / CV_PI * (theta+CV_PI);	//	180deg�𓊉e���̐��Ƀ}�b�s���O
			//	�P����FFT�̌��ʉ摜����T���v��
			double re = cvutil::sampleSubPix(complexDFTPlanes_sth_re, p_sth);
			double im = cvutil::sampleSubPix(complexDFTPlanes_sth_im, p_sth);
			complexDFTImage_uv.at<Vec2d>(i, j) = Vec2d(re, im);		//	2����FFT��(i,j)�v�f�ɃT���v�����ʂ���
		}
	}
	//	DFT���ʕ\���p�Ƀp���[�X�y�N�g���摜�ɕϊ�
	//	���f����L2�m�����̑ΐ�
	Mat complexDFTPlanes_uv[] = { Mat::zeros(optimalDFTSize_uv, CV_64F), Mat::zeros(optimalDFTSize_uv, CV_64F) };
	split(complexDFTImage_uv, complexDFTPlanes_uv);
	Mat magImg_uv;
	magnitude(complexDFTPlanes_uv[0], complexDFTPlanes_uv[1], magImg_uv);		//	���f���v�f�̃m�������i�[
	magImg_uv += Scalar::all(1);	//	�ΐ����̂��߂̃I�t�Z�b�g
	log(magImg_uv, magImg_uv);		//	�l��ΐ���
	//	DFT�摜�͎l��������g�ɂȂ��ďo�͂����̂ŁC�����𒼗������Ɍ����邽�߂ɓ���ւ���
	normalize(magImg_uv, magImg_uv, 0, 1, CV_MINMAX);
	imshow("���e�����琶�������Q����FFT", magImg_uv);

	//	3.3 (u,v)���ʉ摜��2�����t�t�[���G�ϊ����Č��摜�𓾂�
	Mat complexInvDFTImage_xy;
	cvutil::fftShift(complexDFTImage_uv, complexDFTImage_uv);
	dft(complexDFTImage_uv, complexInvDFTImage_xy, DFT_INVERSE + DFT_SCALE);
	Mat complexInvDFTPlanes_xy[]= { Mat::zeros(optimalDFTSize_uv, CV_64F), Mat::zeros(optimalDFTSize_uv, CV_64F) };
	split(complexInvDFTImage_xy, complexInvDFTPlanes_xy);
	Mat invDFTImage_scaled;
	normalize(complexInvDFTPlanes_xy[0], invDFTImage_scaled, 0, 255, CV_MINMAX);
	invDFTImage_scaled.convertTo(invDFTImage_scaled, CV_8UC1);
	imshow("���e����������", invDFTImage_scaled);
	waitKey();

	//---------------------------------------------------------------------------------------------
	//	4. ���e������̍č\���i�t�B���^�␳�t���e�@�j
	//	���g���̈�ł̐ς���ԗ̈�ł̏􍞂݉��Z�Ƃ��ċL�q�\�Ȃ��Ƃ𗘗p����D
	//	�t�[���G�ϊ��@�ɂ����铊�e���̂P�����t�[���G�ϊ��́C���e���ɍ��拭���t�B���^|s|����ݍ��މ��Z��
	//	���w�I�ɓ��l�Ȃ̂ŁC�t�B���^|s|����ݍ��񂾓��e���𕽖�xy�̊p�xth��ɕ��ׂĂ���
	//	���摜������xy�ɕ��������D
	//---------------------------------------------------------------------------------------------
	return 0;
}
