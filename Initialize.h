#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include "DenseTrackStab.h"
#include <getopt.h>

void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap)
{
	trackInfo->length = track_length;
	trackInfo->gap = init_gap;
}

DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;

	long size = height*width*nBins;
	descMat->desc = (float*)malloc(size*sizeof(float));
	memset(descMat->desc, 0, size*sizeof(float));
	return descMat;
}

void ReleDescMat(DescMat* descMat)
{
	if(descMat != NULL) {
		if(descMat->desc != NULL)
			free(descMat->desc);

		free(descMat);
	}


}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->isHof = isHof;
	descInfo->nxCells = nxy_cell;
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;
	descInfo->dim = nBins*nxy_cell*nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}

void InitSeqInfo(SeqInfo* seqInfo, char* video)
{
	VideoCapture capture;
	capture.open(video);

	if(!capture.isOpened())
		fprintf(stderr, "Could not initialize capturing..\n");

	// get the number of frames in the video
	int frame_num = 0;
	while(true) {
		Mat frame;
		capture >> frame;

		if(frame.empty())
			break;

		if(frame_num == 0) {
			seqInfo->width = frame.cols;
			seqInfo->height = frame.rows;
		}

		frame_num++;
    }
	seqInfo->length = frame_num;
}

void usage()
{
	fprintf(stderr, "Extract improved trajectories from a video\n\n");
	fprintf(stderr, "Usage: DenseTrackStab video_file [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");
	fprintf(stderr, "  -v [verbose]              Print descriptor vectors \n");
	fprintf(stderr, "  -p [disable video]        Do not launch video player \n");
	fprintf(stderr, "  -x [HOG]                  Compute histogram of oriented gradients (default: true)\n");
	fprintf(stderr, "  -y [HOF]                  Compute histogram of optical flow (default: true)\n");
	fprintf(stderr, "  -z [MBH]                  Compute motion boundary histograms (default: true)\n");
}

bool arg_parse(int argc, char** argv)
{
	int c;
	bool flag = false;
	bool single_option = true;
	char* executable = basename(argv[0]);
	while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:H:pvxyz")) != -1)
	{
		switch(c) {
			case 'S':
				start_frame = atoi(optarg);
				flag = true;
				break;
			case 'E':
				end_frame = atoi(optarg);
				flag = true;
				break;
			case 'L':
				track_length = atoi(optarg);
				break;
			case 'W':
				min_distance = atoi(optarg);
				break;
			case 'N':
				patch_size = atoi(optarg);
				break;
			case 's':
				nxy_cell = atoi(optarg);
				break;
			case 't':
				nt_cell = atoi(optarg);
				break;
			case 'A':
				scale_num = atoi(optarg);
				break;
			case 'I':
				init_gap = atoi(optarg);
				break;
			case 'h':
				usage();
				exit(0);
				break;
			case 'p':
				show_track = false;
				break;
			case 'v':
				verbose_flag = true;
				break;
			case 'x':
				if(single_option) {
					HOF_flag = false;
					MBH_flag = false;
				}
				HOG_flag = true;
				single_option = false;
				break;
			case 'y':
				if(single_option) {
					HOG_flag = false;
					MBH_flag = false;
				}
				HOF_flag = true;
				single_option = false;
				break;
			case 'z':
				if(single_option) {
					HOG_flag = false;
					HOF_flag = false;
				}
				MBH_flag = true;
				single_option = false;
				break;
			default:
				fprintf(stderr, "error parsing arguments at -%c\n  Try '%s -h' for help.", c, executable );
				abort();
		}
	}
	return flag;
}

#endif /*INITIALIZE_H_*/
