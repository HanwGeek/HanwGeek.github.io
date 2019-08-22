#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <unistd.h>

#ifndef BAND
#define BAND 6
#endif

#ifndef KLAZZ
#define KLAZZ 4
#endif

#ifndef LOOP
#define LOOP 15
#endif

const int h_color[5][3] = {
    {205, 197, 191},//seashell
    {238, 118, 0},//darkorange2
    {50, 205, 50},//limegreen
    {135, 206, 250},//lightskyblue
    {168, 30, 199}//purple
};

cv::Mat h_img[BAND];
cv::Mat h_show;

int row;
int col;
int opt;
int size, rank;
int imgSize;
int totalCount;
int count;
int isStop;
int **h_center;
int **h_oldCenter;
int *h_count;
int *feature;
int *h_feature;

unsigned char **data;

clock_t start, end;
clock_t read_start, read_end;
clock_t write_start, write_end;
clock_t scatter_start, scatter_end;

int loadImage(std::string infoName);
int updateSample(int pos);
int check();
void imgShow();
void initial();
void initMain();
void calcTime();
void initChild();
void updateCenter();

int main(int argc, char* argv[])
{
     std::string infoName;
    if (rank == 0) 
    {
    	start = clock();
    	while ((opt = getopt(argc, argv, "p:")) != -1)
    	{
    		switch(opt)
    		{
    			case 'p': infoName = std::string(optarg);
    			break;
    		}
    	
    	}
    }
    //std::string infoName("./image1.info");

    MPI_Status status;

    //Initialize MPI environment
    MPI_Init(&argc, &argv);    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Initialize center of all process
    initial();
    
    //Initial main process
    if (rank == 0)
    {
    	read_start = clock();
        loadImage(infoName);
        read_end = clock();
        initMain();
    }

    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Initial child process
    initChild();
    
    //Scatter image
    scatter_start = clock();
    for (int i = 0; i < BAND; i++)
    {
        MPI_Scatter(h_img[i].data, count, MPI_UNSIGNED_CHAR, 
            data[i], count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);  
    }
    scatter_end = clock();
    //Start loop
    for (int k = 0; k < LOOP; k++)
    {
        //Broadcast center infomation
        for (int i = 0; i < KLAZZ; i++)
        {
            MPI_Bcast(h_center[i], BAND, MPI_INT, 0, MPI_COMM_WORLD);
        }
            
        //Child process calculate
        for (int i = 0; i < count; i++)
        {
            updateSample(i);
        }

        //Update center infomation
        
        MPI_Gather(feature, count, MPI_INT,
            h_feature, count, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0)
        {
            updateCenter();
        }
        //Check
        // if (rank == 0) 
        // {
        //     //print output
        //     //imgShow();
        //     isStop = check();
        // }
        // MPI_Bcast(&isStop, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // if (isStop == 1)
        // {
        // 	break;
        // }
    }
    if (rank == 0) 
    	{
    		write_start = clock(); 
    		imgShow();
    		write_end = clock();
    	}

    MPI_Finalize();

	if (rank == 0)
	{
		end = clock();
		calcTime();
	}
    return 0;
}

int loadImage(std::string infoName)
{
	std::string infoFileName = infoName;
    std::fstream infoFile(infoFileName.c_str());
    if (!infoFile.is_open())
    {
        std::cout << "Fail to open info file!";
        return 0;
    }
    std::string imgName;
    for (int i = 0; i < BAND; i++)
    {
		infoFile >> imgName;
		std::string imgFilePath = infoName + imgName;
    	h_img[i] = cv::imread(imgFilePath, 0);				
    }

    return 1;
}

void initial()
{
    h_center = new int*[KLAZZ];
    h_count = new int[KLAZZ];
    for (int i = 0; i < KLAZZ; i++)
    {
        h_center[i] = new int[BAND];
        h_count[i] = 0;
    }
}

void initMain()
{
    row = h_img[0].rows;
    col = h_img[0].cols;
    totalCount = row*col;
    count = totalCount/size;
    h_oldCenter = new int*[KLAZZ];
    for (int i = 0; i < KLAZZ; i++)
    {
        h_oldCenter[i] = new int[BAND];
    }
    h_feature = new int [totalCount];
    h_show = cv::Mat::zeros(row, col, CV_8UC3);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < KLAZZ; i++)
    {
        for (int j = 0; j < BAND; j++)
        {
            h_center[i][j] = rand() % 50 + 20;
            h_oldCenter[i][j] = h_center[i][j];
        }
    }
}

void initChild()
{
    data = new unsigned char*[BAND];
    for (int i = 0; i < BAND; i++)
    {
        data[i] = new unsigned char[count]; 
    }
    feature = new int[count];
}

inline int updateSample(int pos)
{
    int idx = 0;
    int min = INT_MAX;

    int s[6];
    for (int i = 0; i < BAND; i++)
    {
        s[i] = (int)data[i][pos];
    }  

    for (int i = 0; i < KLAZZ; i++)
    {
        int distance = 0;
        for (int j = 0; j < BAND; j++)
        {
            distance += (h_center[i][j] - s[j]) * (h_center[i][j] - s[j]);
        }

        if (distance < min)
        {
            min = distance;
            idx = i;
        }
    }


    feature[pos] = idx;
    return idx;
}

void updateCenter()
{
    for (int i = 0; i < totalCount; i++)
    {
        int idx = (int)h_feature[i];
        h_count[idx]++;
    }
    for (int i = 0; i < KLAZZ; i++)
    {
        if (h_count[i] != 0)
        {
            for (int j = 0; j < BAND; j++)
            {
                h_oldCenter[i][j] = h_center[i][j];
                h_center[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < totalCount; i++)
    {
        int idx = h_feature[i];
        for (int j = 0; j < BAND; j++)
        {
            h_center[idx][j] += (int)*(h_img[j].data + i);
        }
    }

    for (int i = 0; i < KLAZZ; i++)
    {
        for (int j = 0; j < BAND; j++)
        {   
            if (h_count[i] != 0)
            {
                h_center[i][j] = h_center[i][j] / h_count[i];
            }
        }
    }
    for (int i = 0; i < KLAZZ; i++)
    {
        h_count[i] = 0;
    }
}

int check()
{
    for (int i = 0; i < KLAZZ; i++)
    {
        for (int j = 0; j < BAND; j++)
        {
            int eps = h_center[i][j] - h_oldCenter[i][j];
            if (eps < -1 || eps > 1)
            {
                return 0;
            }
        }
    }
    return 1;
}

void imgShow()
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int idx = h_feature[i * col + j];
            h_show.at<cv::Vec3b>(i, j) = cv::Vec3b(h_color[idx][0], h_color[idx][1], h_color[idx][2]);          
        }
    }
    cv::imwrite("pResult.TIF", h_show);  
}

void calcTime()
{
	double read_time = (double)(read_end - read_start)/CLOCKS_PER_SEC;
 	double write_time = (double)(write_end - write_start)/CLOCKS_PER_SEC;
 	double scatter_time = (double)(scatter_end - scatter_start)/CLOCKS_PER_SEC;
 	double total_time = (double)(end - start)/CLOCKS_PER_SEC;
 	std::cout << "Read time:" << read_time << std::endl;
 	std::cout << "Write time:" << write_time << std::endl;
 	std::cout << "Scatter time:" << scatter_time << std::endl;
 	std::cout << "Total time:" << total_time << std::endl;
}