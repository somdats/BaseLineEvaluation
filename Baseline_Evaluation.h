#pragma once
// WinAPI
//#include <windows.h>

// WIndows header
#include<string>
#include<vector>

// Eigen headers
#include<Eigen/core>
#include<Eigen/Geometry>

//pcl headers
#include<pcl/Vertices.h>
#include<pcl/PolygonMesh.h>
#include<pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>

//pcp headers
#include"Datatypes.h"
#include"Evaluations_ETH.h"
#include"CorrespondenceEstimationMLS.h"
#include"CpuTimeProfiler.h"
#include "gicp.h"
#include"app.h"
#include"pct_io.h"
#include"CorrespondenceEstimationDepthMap.h"
#include"CorrespondenceEstimationPlaneProjection.h"
#include"PlaneProjectionMethod.h"
#include"InterpolationMethod.h"


#define NOMINMAX 1
#define FGR_BINARY "C:/Softwares/FGR/source/Build/FastGlobalRegistration/Release/FastGlobalRegistration.exe"
using namespace std;
using namespace dgc::gicp;


namespace Baseline
{
    class BaseLine
    {
    public:
        BaseLine(int numsamples,double offset = 1.0,double scale_radius = 5.0,bool eth_d = false) :
            m_meanMeshRes(0.0),
            Error_StdDev(1e-7f),
            StopCriteria(1e-7f),
            m_FileExtension("ply"),
            m_SampleSize(numsamples),
            m_searchRadius(scale_radius),
            m_Threshold(5.0),
            offset(offset),
            eth_evaluation(eth_d)
        {

        };

        typedef PointData<float> PointDataType;
        virtual void Evaluate() = 0;
        void BasefolderPath(const string &DirName);
        void SetDataPath(const string &dirName);
        void  MeshDataPath(const string &dirName); // for  meshresolution computation
        // Output Paths
        void SaveRMSE(const string FileName);
        void SaveViewPair(const string FileName);
        void MergeFileLocation(const string &dirName);
        void SaveTransformationMatrices(string FileName);
        void PrepareEvaluation();
        void EvaluateData();
        void WriteAndPrintResults(string &MetricFileName,string &RmseFileName);
        void SetParamters(double std_dev, int numSamples);
        double ComputeStopCriteria(double OffSet, float unit, double std_dev);
        void ReadInputCloudForEvaluation(string &inputSourceFile, string &inputTargetFile, int index, 
            CloudWithoutType &CloudSource, CloudWithoutType &CloudTarget);
       void SetOutPutDirectoryPath(const string &DirName);
       void ReadFolderNameFromList(string &FileName);
       double GetTotalCPUTime();
       void WriteProcessingTime(string &FileName);
       void WriteCorrespondenceAndOptimizationTime(string &FileName);
       void Reset();
       void ExtractStandardDeviation(double &Value, string fileName);
       void SetAlgorithmName(string &AlName);
       void WriteAllRMSEForAMethod(const string & FileName);
       void ReadRMSEFromAFile( string & FileName);
       // simulated scans evaluation function
       void PrepareEvaluationOFSimulatedScans();
       void GenerateSimulatedScanViewPair(const std::string &dirEvalName, const std::string &dirGroundTruthName, const std::string &fileExt); // generates view pair for evaluation & groundtruth rmse computation
       void EvaluateAllViewPairForSimulatedScans();
       virtual double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud, string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
           double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity()) = 0;
       void SetGroundTruthFolderNames(const string &DirName);
       void setOutPutTransformationFileName(const string &FilerName);
       void setProcessingTimeFileName(const string &FilerName);
       void PrepareEvaluationOfEthData();
       double ComputeRMSEForEthDataset(CloudWithoutType cloudA, CloudWithoutType cloudB);
       void GenerateRMSEForExternalMethod();
       bool GeneratePctViewPairs(const std::string &dirEvalName, const std::string &dirGroundTruthName, const std::string &fileExt);
       std::vector<UVData<float>> LoadPCTFiles(int fileCounter);
       void SetInverseCalibrationMethod(bool option);
       Eigen::Matrix4f LoadProjectionMatrix(string &fileNameA, int folderIndex);
       void ReadRegistrationTime(string &fileName, std:: vector<double>&correstime, std::vector<double>&optim_time, std::vector<double>&regis_time);
       void ReadRegistrationTimeAndRMSEForAllDataset(string & ourceFileA, string &sourceFileB,
           std::vector<double>&rmse_avg, std::vector<double>&time_avg);

       void CreateIndicesVsPointMap(CloudWithoutType & PointCloud);
      
       string SplitFolderNameFromPath(string &FolderWithPath);
       void ConvertPCDToMesh(CloudWithoutType &inputCloud, pcl::PolygonMesh &mesh);

       void RegisterMultipleData();
       void RegisterMultipleDataPairwise();
       void GeneratePairWiseViewPair(const std::string &dirEvalName, const std::string &fileExt);
       void TransformSuccessiveViewPair(std::string transformfileName);
       CloudWithoutType FilterPlaneFromPointCloud(CloudWithoutType &inputCloud);
       void FilterCloud();
       void LoadPCDAndSavePly(const std::string &dirEvalName, const std::string &fileExt);


     

    protected:
        string m_BaseFolderPath;
        string  m_DataSetPath;
        string  m_MeshDataPath;
        string m_FileExtension;
        string m_MethodName;
        string m_RmseFileName;
        string m_OutputMergedDir;
        string m_transformFileName;
        string m_processingTimeFileName;
        string m_newFolderPath;
        string m_pctExtension = "pct";
        

        double			m_meanMeshRes;
        double          Error_StdDev;
        double          StopCriteria;
        double         SuccessRate;
        int            m_SampleSize;
        float          m_searchRadius;
        double         m_Threshold;
        double        m_TotalTime;
        double        offset;
        double m_diagonalLength;

        bool inverse_calibration_method = false;
        bool eth_evaluation;
        std::vector<string> m_groundTruthFileList;
        std::vector<string>m_DataFolderList;
        std::vector<string>	m_vMeshAbsFileNames;
        std::vector<std::pair<std::string, std::string>> m_ViewPairs;  // for LRF protocol Evaluation
        std::vector<std::pair<std::string, Eigen::Matrix4f>> View_With_GroundTruth;
        std::vector<Eigen::Affine3f> initialPoses;
        std::vector<double> RMSEs;
        std::vector<Eigen::Matrix4f> m_OutPutTransformations;
        std::vector<double>	m_vPairwiseCpuTimes;
        std::vector<CloudWithoutType> m_Clouds;
        std::vector<CloudWithoutType> m_TransformedClouds;
        std::vector<double>m_samplingTime;
        std::vector<double>m_smoothingTime;
        std::vector<double>m_registration_time;
        std::vector<double>m_corresp_time;
        std::vector<double>optim_time;
        std::vector<std::pair<std::string, std::string>>m_SimulatedViewPairs;
        std::vector<std::pair<std::string, std::string>>m_SimulatedGroundTruthViewPairs;
        std::vector<std::pair<std::string, std::string>>m_SimulatedPctViews;
        std::vector<Eigen::Matrix4f> GroundTruthPoses; // ETH Dataset
        std::vector<Eigen::Matrix4f>InitialPose_Eth; // ETH Dataset
        std::vector < std::pair<double, double>> Rot_Trans_Error; // ETH Dataset
        std::vector <UVData<float>>m_TargetPixels;

        std::vector<double>featureSource;
        std::vector<double>featureTarget;

        std::vector<Eigen::Matrix3f>src_principal_frame;
        std::vector<Eigen::Matrix3f>tgt_principal_frame;
        std::vector<Eigen::Vector3f> source_eig_value;
        std::vector<Eigen::Vector3f>tar_eig_value;
        std::map<PointDataType, int>Principal_frame_Index_source_map;

        Eigen::Matrix4f m_projectionMatrix;
        void SortData(std::vector<double> &RMSEs)
        {
            typedef double dbl;
            std::sort(RMSEs.begin(), RMSEs.end(),
                [&](const dbl& v1, const dbl& v2)
            {
                return v1 < v2;
            });
        }

        Evaluation_Ext  LRF_Protocols;
        Evaluations   ETH_Protocols;

        CloudWithoutType m_sourceGroundTruth;
        CloudWithoutType m_targetGroundTruth;
        CloudWithoutType m_sampledSourceCloud;
       

    };

    class BaseLine_WithTargetSmoothing : public BaseLine
    {
    public:
        BaseLine_WithTargetSmoothing( int numsamples,/*string dirName = "",*/ bool WithNormalSmoothing =true,
            double multiplier = 5.0, double scale_radius = 5.0, bool eth_d = false) :
           /* m_DirName(dirName),*/
            m_WithNormalSmoothing(WithNormalSmoothing),
            stop_criteria_multiplier(multiplier),
            BaseLine(numsamples,multiplier, scale_radius, eth_d)
        {

        };
        
        void Evaluate();
        void ComputeNormalSmoothingUsingMLSProjection(std::vector<std::pair<std::string, std::string>> &views,
            string &inputFileName, int index, CloudWithoutType &inputCloud, cMLSCorrespondence &mlsCorres,
            CloudWithoutType &SmoothedNormalCloud);
         double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,  string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
            double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());
        

    protected:
        string m_DirName;   // newdirectory for saving merged file
        double normal_filtering_threshold;
        double stop_criteria_multiplier;
        bool m_WithNormalSmoothing;
      
    };

}

class NearestNeigborSicp :public Baseline::BaseLine
{
public:
    NearestNeigborSicp(/*string dirName = "",*/ int numsamples, bool fallback = false, double multiplier = 5.0,  bool eth_d = false) :
        /* m_DirName(dirName),*/
        stop_criteria_multiplier(multiplier),
        m_Fallback(fallback),
        BaseLine(numsamples,multiplier, 10.0, eth_d)
    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());



protected:
    string m_DirName;   // newdirectory for saving merged file
    double stop_criteria_multiplier;
    bool m_Fallback;

};
class MLSWithFilter : public Baseline::BaseLine
{
public:
    MLSWithFilter( int numsamples,double normal_threshold, bool fallback = true, double multiplier = 5.0, double radius_scale = 0.5, bool eth_d = false) :

        stop_criteria_multiplier(multiplier),
        m_normThreshold(normal_threshold),
        m_RadiusScale(radius_scale),
        m_Fallback(fallback),
        BaseLine(numsamples,multiplier, radius_scale, eth_d)

    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());
    void SmoothTargetUsingMLSProjection(std::vector<std::pair<std::string, std::string>> &views,
        string &inputFileName, int index, CloudWithoutType &inputCloud, cMLSCorrespondence &mlsCorres,
        CloudWithoutType &SmoothedNormalCloud);


protected:
    double stop_criteria_multiplier;
    bool m_Fallback;
    double m_normThreshold;
    double m_RadiusScale;
};
class GICPPairwiseRegistration : public Baseline::BaseLine
{
protected:
    int								m_numMaxIter;
    double							m_epsilon;
    float							m_maxSearchRadius;

public:
    GICPPairwiseRegistration(int numsamples, int numMaxIter = 5, double epsilon = 0.0001, float maxSearchRadius = 0.2, float radius_scale = 5.0f, bool eth_data = false) :
        m_numMaxIter(numMaxIter),
        m_epsilon(epsilon),
        m_maxSearchRadius(maxSearchRadius),
        BaseLine(numsamples, 5.0, radius_scale, eth_data)
    {

    };
    void SetNumMaxIter(int numMaxIter) { m_numMaxIter = numMaxIter; }
    void SetEpsilon(double epsilon) { m_epsilon = epsilon; }
    void SetMaxSearchRadius(double maxSearchRadius) { m_maxSearchRadius = maxSearchRadius; }
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());
};
class TwoStepBaseline : public Baseline::BaseLine
{
public:
    TwoStepBaseline(int numsamples, double normal_threshold, bool fallback = true, double multiplier = 5.0, double radius_scale = 0.5, bool eth_d = false) :

        stop_criteria_multiplier(multiplier),
        m_normThreshold(normal_threshold),
        m_RadiusScale(radius_scale),
        m_Fallback(fallback),
        BaseLine(numsamples, multiplier, radius_scale, eth_d)

    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());


protected:
    double stop_criteria_multiplier;
    bool m_Fallback;
    double m_normThreshold;
    double m_RadiusScale;
};

class FGRPairWiseRegistration : public Baseline::BaseLine
{
public:
    FGRPairWiseRegistration(int numsamples, double radius_scale, bool eth_d) :
        max_Radius(radius_scale),
        BaseLine(numsamples, 5.0, radius_scale, eth_d)
    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());

protected:
    double max_Radius;
};

class WeightedDepthProjection : public Baseline::BaseLine
{
public:
    WeightedDepthProjection(/*string &ProjectionMatrixName,*/ int numsamples) :
        /*ProjectionMatrix(tool::LoadProjectionmatrix(ProjectionMatrixName)),*/
        BaseLine(numsamples, 5.0, 5.0, false)
    {

    };
    
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());
protected:
    Eigen::Matrix4f ProjectionMatrix;
};

class WeightedPlaneProjection : public Baseline::BaseLine
{
public:
    WeightedPlaneProjection(/*string &ProjectionMatrixName,*/int numsamples) :
        /*ProjectionMatrix(tool::LoadProjectionmatrix(ProjectionMatrixName)),*/
        BaseLine(numsamples, 5.0, 5.0, false)
    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());

protected:
    Eigen::Matrix4f ProjectionMatrix;
};

class CurvatureBasedRegistration : public Baseline::BaseLine
{
public:
    CurvatureBasedRegistration(int numsamples, int nnquery, double radiuscale) :
        nn_query(nnquery),
        m_RadiusScale(radiuscale),
        BaseLine(numsamples, 5.0, radiuscale, false)
    {

    };
    void Evaluate();
    double PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
        string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
        double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix = Eigen::Affine3f::Identity());
    void ComputeLocalPrincipalFrame(string &inputTargetFileName,string &inputFileName, int index, CloudWithoutType &inputSourceCloud, CloudWithoutType &inputtargetCloud, std::vector<Eigen::Vector3f>&egv_s,
        std::vector<Eigen::Vector3f>&egv_t);
protected:
   int nn_query;
   double m_RadiusScale;
  

 };