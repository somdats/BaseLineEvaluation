#include"Baseline_Evaluation.h"
#include "ErrorMetric.h"
#include"Transformation_tool.h"
#include<pcl/common/transforms.h>
#include<pcl/io/ply_io.h>
#include<pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/filters/normal_space.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include<pcl/surface/gp3.h>
#include"TransformationEstimationSparseICP.h"
#include"ply_file_io.h"
#include"Datatypes.h"
#include"Transformation_tool.h"
#include"pct_io.h"
#include"NormalSpaceSampling.h"
#include"KdTreeSearch.h"
#include"wfuncs.h"
#include"MLSSearch.h"
#include"NormalEstimationPoints.h"
#include"Common.h"
#include "FPFHfeatures.h"
#include <cstdio>

 void Baseline::BaseLine:: Reset()
{
    StopCriteria = 1e-7f;
    Error_StdDev = 1e-7f;
    m_vMeshAbsFileNames.clear();
    m_ViewPairs.clear();
    View_With_GroundTruth.clear();
    initialPoses.clear();
    RMSEs.clear();
    m_OutPutTransformations.clear();
    m_vPairwiseCpuTimes.clear();
    m_samplingTime.clear();
    m_smoothingTime.clear();
    m_registration_time.clear();
    GroundTruthPoses.clear();
    InitialPose_Eth.clear();


}
void Baseline::BaseLine::BasefolderPath(const string &DirName)
{
    m_BaseFolderPath = DirName;
   
}
void Baseline::BaseLine::SetAlgorithmName(string &AlName)
{
    m_MethodName = AlName;
}
void Baseline::BaseLine::SetDataPath(const string &dirName)
{
    m_DataSetPath = dirName;
}

void Baseline::BaseLine::MeshDataPath(const string &dirName)
{
    m_MeshDataPath = dirName;
 }

void Baseline::BaseLine:: SetOutPutDirectoryPath(const string &DirName)
{
    m_OutputMergedDir = DirName;
}
void Baseline::BaseLine::setOutPutTransformationFileName(const string &FileName)
{
    m_transformFileName = FileName;
}
void Baseline::BaseLine::setProcessingTimeFileName(const string &FileName)
{
    m_processingTimeFileName = FileName;
}

void Baseline::BaseLine::SaveRMSE(const string FileName)
{
    if (m_DataSetPath == "")
    {
        throw runtime_error("ERROR: absDatasetPath not set. Use SetDataPath() for setting the path containing partial views");
    }
    std::string FullLocation = m_DataSetPath + "/" + this->m_OutputMergedDir + "/"  + FileName; //+ this->m_OutputMergedDir + "/"
    if (iospace::ExistsFile(FullLocation))
    {
        remove(FullLocation.c_str());
    }
    metric::WriteRMSE(FullLocation, RMSEs);
 }

void Baseline::BaseLine::SaveViewPair(const string dirName)
{
    if (m_DataSetPath == "")
    {
        throw runtime_error("ERROR: absDatasetPath not set. Use SetDataPath() for setting the path containing partial views");
    }
    std::string FulldirName = m_DataSetPath + "/" + dirName;
    LRF_Protocols.WriteViewpair(m_ViewPairs, FulldirName);

}
void Baseline::BaseLine:: MergeFileLocation(const string &dirName)
{
    if (m_DataSetPath == "")
    {
        throw runtime_error("ERROR: absDatasetPath not set. Use SetDataPath() for setting the path containing partial views");
    }
}
double Baseline::BaseLine::ComputeRMSEForEthDataset(CloudWithoutType cloudA, CloudWithoutType cloudB)
{
    int nCount = 0;
    float dError = 0.0f;
    CloudPtr pTarget(new pcl::PointCloud <PointType>);
    CloudPtr pSource(new pcl::PointCloud <PointType>);
    pcl::fromPCLPointCloud2(*cloudA, *pSource);
    pcl::fromPCLPointCloud2(*cloudB, *pTarget);
    if (pSource->points.size() != pTarget->points.size())
    {
        return -1.0;
    }
    for (int i = 0; i < pSource->points.size(); i++)
    {
        dError += (pSource->points[i].getVector3fMap() - pTarget->points[i].getVector3fMap()).squaredNorm();
        nCount++;
    }
  
    double rmse_unit = metric::EstimatePointAvgDistance(cloudA);
    dError = (dError) / (float)nCount;
    dError = dError / (static_cast<float>(rmse_unit) * static_cast<float>(rmse_unit));
    dError = sqrt(dError);
    return dError;
}
void Baseline::BaseLine::SaveTransformationMatrices(string FileName)
{
    if (m_DataSetPath == "")
    {
        throw runtime_error("ERROR: absDatasetPath not set. Use SetDataPath() for setting the path containing partial views");
    }
    std::string fullLocation = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + FileName; //m_DataSetPath
    if (iospace::ExistsFile(fullLocation))
    {
        remove(fullLocation.c_str());
    }

    if (m_OutPutTransformations.size() == 0)
    {
        throw runtime_error(" OutputTransformation Matrices size is zero");
    }

    tool::writeTransformationMatrix(fullLocation, m_OutPutTransformations);

}
void Baseline::BaseLine::ReadRMSEFromAFile(string & FileName)
{
    std::string line;
    std::ifstream myfile(FileName.c_str());
    RMSEs.clear();
    RMSEs.reserve(100000);
    unsigned numberOfLines = 0;
    if (myfile.is_open())
    {
        while (std::getline(myfile, line))
        {
            double rmse = atof(line.c_str());
            if (rmse != -1.0 && rmse < m_Threshold)  // reads only value which is lower than threshold for evaluation puropose
            {
                RMSEs.push_back(rmse);
                numberOfLines++;
            }
        }
     
    }
    std::cout << "  CorrectRMSEs:" << numberOfLines << std::endl;
    myfile.close();
}

void Baseline::BaseLine::WriteAllRMSEForAMethod(const string & FileName)
{
    int numFolder = m_DataFolderList.size();
    if (numFolder == 0)
    {
        throw runtime_error("ERROR: No Data to Evaluate\n");
    }
    std::string OutPutDirectory = m_BaseFolderPath + "/" + "DataSetRMSE";
    string fullFileName = OutPutDirectory + "/" + m_MethodName + "_rmse_" + FileName;
    if (iospace::ExistsFile(fullFileName))
    {
        remove(fullFileName.c_str());
     }
    std::vector<double> rmse_per_dataset;
    for (int ItrFolder = 0; ItrFolder < numFolder; ItrFolder++)
    {
        std::cout << " RMSE Processing:" << ItrFolder + 1 << "/" << numFolder << std::endl;
        string FolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder];
        std::string rmse_filename = FolderPath +"/" + m_DataFolderList[ItrFolder] + "_rmse_Baseline_" + m_MethodName + ".txt";
        std::cout << " FolderName:" << m_DataFolderList[ItrFolder];
        ReadRMSEFromAFile(rmse_filename);
     
        std::move(RMSEs.begin(), RMSEs.end(), std::back_inserter(rmse_per_dataset));
        
    }
    SortData(rmse_per_dataset);
    metric::WriteRMSE(fullFileName, rmse_per_dataset);
}
void Baseline::BaseLine::PrepareEvaluation()   // for LRF protocols
{
    // compute MeshResolution
    m_meanMeshRes =  LRF_Protocols.ComputeMeshResolutionForAllViews(m_MeshDataPath, m_FileExtension, m_DataSetPath + "/MeshRes.txt");
    m_ViewPairs =  LRF_Protocols.GenerateViewPairForEvaluation(m_DataSetPath, m_FileExtension);
    View_With_GroundTruth = LRF_Protocols.ReadTransformationmatrixFromFile(m_MeshDataPath + "/groundtruth.txt");
    initialPoses =  tool::ReadTransformationMatrix(m_MeshDataPath + "/coarse_registration.txt");
    m_OutPutTransformations.resize(m_ViewPairs.size());
    m_vPairwiseCpuTimes.resize(m_ViewPairs.size(), 0.0);
    RMSEs.resize(m_ViewPairs.size(), 0.0);
    m_samplingTime.resize(m_ViewPairs.size(), 0.0);
    m_smoothingTime.resize(m_ViewPairs.size(), 0.0);
    m_registration_time.resize(m_ViewPairs.size(), 0.0);
    m_corresp_time.resize(m_ViewPairs.size(), 0.0);
    optim_time.resize(m_ViewPairs.size(), 0.0);
}
void Baseline::BaseLine::PrepareEvaluationOfEthData()
{
    string ProtocolFileName = m_DataSetPath + "/" + "eth_protocol.csv";
    string ValidationFileName = m_DataSetPath + "/" + "eth_validation.csv";
    ETH_Protocols.ParseProtocolFile(ProtocolFileName);
    ETH_Protocols.ParseValidationFile(ValidationFileName);
    m_ViewPairs = ETH_Protocols.GetComparisonDataName();
    InitialPose_Eth = ETH_Protocols.GetPerturbationData();
    GroundTruthPoses = ETH_Protocols.GetGroundTruthData();
    m_OutPutTransformations.resize(m_ViewPairs.size());
    m_vPairwiseCpuTimes.resize(m_ViewPairs.size(), 0.0);
    m_samplingTime.resize(m_ViewPairs.size(), 0.0);
    m_smoothingTime.resize(m_ViewPairs.size(), 0.0);
    m_registration_time.resize(m_ViewPairs.size(), 0.0);
    m_corresp_time.resize(m_ViewPairs.size(), 0.0);
    optim_time.resize(m_ViewPairs.size(), 0.0);

    RMSEs.resize(m_ViewPairs.size(), 0.0);
    Rot_Trans_Error.resize(m_ViewPairs.size(), std::pair<double, double>(0.0, 0.0));

}
void Baseline::BaseLine::WriteAndPrintResults(string &MetricFileName, string &RmseFileName)
{
    int trueCount, FalseCount;
    if (m_DataSetPath == "")
    {
        throw runtime_error("ERROR: absDatasetPath not set. Use SetDataPath() for setting the path containing partial views");
    }
    m_RmseFileName = RmseFileName;
    SaveRMSE(m_RmseFileName);
    string FullFileName = m_DataSetPath + "/" + m_RmseFileName;
    LRF_Protocols.CompareRMSEWithThreshold(FullFileName, m_Threshold, trueCount, FalseCount);
    std::cout << "TrueCount:" << trueCount << std::endl;
    std::cout << "falseCount:" << FalseCount << std::endl;
    string fileName  = m_DataSetPath + "/" + MetricFileName;
    LRF_Protocols.PrintResults(fileName, Error_StdDev, m_meanMeshRes, m_ViewPairs.size(), trueCount);
  
}
void Baseline::BaseLine::EvaluateData()
{
    int numFolder = m_DataFolderList.size();
    if (numFolder == 0)
    {
        throw runtime_error("ERROR: No Data to Evaluate\n");
    }
    for (int ItrFolder = 0; ItrFolder < numFolder; ItrFolder++)
    {
        string FolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder];
        string meshFolderpath = FolderPath + "/" + m_DataFolderList[ItrFolder] + "_mesh";
       // Output paths
        std::string transform_fileName = m_DataFolderList[ItrFolder] + "_pairwise_transform_Baseline_"
            + m_MethodName + ".txt";
        std::string view_pair = m_DataFolderList[ItrFolder] + "_registration_view_pair_Baseline_"
            + m_MethodName + ".txt";
        std::string rmse_filename = m_DataFolderList[ItrFolder] + "_rmse_Baseline_" + m_MethodName + ".txt";
        std::string output_fileName = m_DataFolderList[ItrFolder] + "_performance_metric_Baseline_"
            + m_MethodName + ".txt";
        string processingtime = FolderPath + "/" + m_DataFolderList[ItrFolder] + "_processing_time_Baseline_"
            + m_MethodName + ".txt";
        string icp_component_time = FolderPath + "/" + m_DataFolderList[ItrFolder] + "_icp_component_time_"
            + m_MethodName + ".txt";
        std::string MergeLocation = "Baseline_" + m_MethodName;
        SetDataPath(FolderPath);
        MeshDataPath(meshFolderpath);
        SetOutPutDirectoryPath(MergeLocation);
        std::string newFolder = m_DataFolderList[ItrFolder] + "_new";
        string NewFolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder] + "/" + newFolder;
        m_newFolderPath = NewFolderPath;
        if (false == eth_evaluation)
        {
            PrepareEvaluation();
           // LRF_Protocols.ComputePointNormal(meshFolderpath, "ply", NewFolderPath);  // change it to:FolderPath after the completion of LRF on sicp
            Evaluate();
          // WriteAndPrintResults(output_fileName, rmse_filename);
         
        }
        else
        {
            string fileName = FolderPath + "/" + "error_" + m_DataFolderList[ItrFolder] + "_" + m_MethodName + ".txt";
            PrepareEvaluationOfEthData();
            Evaluate();
            ETH_Protocols.setEvaluationParameter(Rot_Trans_Error);
            ETH_Protocols.CollectFinalRegistrationMatrices(m_OutPutTransformations);
            ETH_Protocols.setComputationTimeforMethod(m_registration_time);
            //ETH_Protocols.WriteEvaluationParameter(fileName);
           // WriteAndPrintResults(output_fileName, rmse_filename);
            ETH_Protocols.Reset();
        }
       // SaveTransformationMatrices(transform_fileName);
       // WriteProcessingTime(processingtime);
       // WriteCorrespondenceAndOptimizationTime(icp_component_time);
        Reset();
        
    }
 }

void Baseline::BaseLine::GenerateRMSEForExternalMethod()
{
    int numFolder = m_DataFolderList.size();
    if (numFolder == 0)
    {
        throw runtime_error("ERROR: No Data to Evaluate\n");
    }
    for (int ItrFolder = 0; ItrFolder < numFolder; ItrFolder++)
    {
        string FolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder];
      
        std::string output_fileName = m_DataFolderList[ItrFolder] + "_performance_metric_Baseline_"
            + m_MethodName + ".txt";
        std::string rmse_filename = m_DataFolderList[ItrFolder] + "_rmse_Baseline_" + m_MethodName + ".txt";
        string methodData = m_DataFolderList[ItrFolder] + "_" + m_MethodName + ".csv";
        SetDataPath(FolderPath);

        string methodFileName = FolderPath + "/" + methodData;
        std::vector<Eigen::Matrix4f> method_results;
        std::vector<double>method_execution_time;
        string ProtocolFileName = m_DataSetPath + "/" + "eth_protocol.csv";
        string ValidationFileName = m_DataSetPath + "/" + "eth_validation.csv";

        ETH_Protocols.ReadMethodComparisonData(methodFileName, method_results, method_execution_time);
        ETH_Protocols.ParseProtocolFile(ProtocolFileName);
        ETH_Protocols.ParseValidationFile(ValidationFileName);
        m_ViewPairs = ETH_Protocols.GetComparisonDataName();
        GroundTruthPoses = ETH_Protocols.GetGroundTruthData();
        RMSEs.resize(m_ViewPairs.size(),0.0);
        string sourceInput = "";
        string targetInput = "";
        string smoothTargetFile = "";
        CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
        CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
        CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);

        for (int i = 0; i < method_results.size(); i++)
        {

            std::cout << "Evaluation pair:" << i + 1 << "/" << method_results.size() << std::endl;
            // ReadViewpair for Evaluation 

            ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
            //// load source ply file

            CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
            CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, method_results[i]);
            RMSEs[i] =(ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed));
        }
        WriteAndPrintResults(output_fileName, rmse_filename);
        Reset();
        ETH_Protocols.Reset();
        m_ViewPairs.shrink_to_fit();
        GroundTruthPoses.shrink_to_fit();
        RMSEs.shrink_to_fit();
       
    }
   
}
void Baseline::BaseLine::ConvertPCDToMesh(CloudWithoutType &inputCloud, pcl::PolygonMesh &mesh)
{
    CloudWithNormalPtr v_n(new pcl::PointCloud <PointNormalType>);
    pcl::fromPCLPointCloud2(*inputCloud, *v_n);
    pcl::search::KdTree<PointNormalType>::Ptr tree(new pcl::search::KdTree<PointNormalType>);
    tree->setInputCloud(v_n);
    pcl::GreedyProjectionTriangulation<PointNormalType>gp;
    pcl::PolygonMesh triangles;
    float avgdist = 10.0 *  metric::EstimatePointAvgDistance(inputCloud);
    gp.setSearchRadius(avgdist);
    // Set typical values for the parameters
    gp.setMu(2.5);  //2.5
    gp.setMaximumNearestNeighbors(100);
    gp.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp.setMinimumAngle(M_PI/6); // 10 degrees
    gp.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp.setNormalConsistency(false);
    gp.setInputCloud(v_n);
    gp.setSearchMethod(tree);
    gp.setConsistentVertexOrdering(false);
    gp.reconstruct(triangles);
    mesh = triangles;
}
void Baseline::BaseLine::TransformSuccessiveViewPair(string transformfileName)
{
    std::string fullLocation = m_DataSetPath + this->m_OutputMergedDir + "/" + transformfileName;
    std::vector<Eigen::Affine3f> tfs_matrix = tool::ReadTransformationMatrix(fullLocation);
    std::vector<Eigen::Matrix4f> combine_matrix;
    combine_matrix.resize(tfs_matrix.size());
    combine_matrix[0] = (Eigen::Matrix4f::Identity());
    if (m_SimulatedViewPairs.size() == 0)
    {
        GeneratePairWiseViewPair(m_DataSetPath, m_FileExtension);
    }
    for (int i = 0; i < tfs_matrix.size(); i++)  //tfs_matrix.size()
    {
        if (i == 0)
            combine_matrix[i] = combine_matrix[i] * tfs_matrix[i].matrix();
      /*  else if (i == tfs_matrix.size() - 1 || i  == tfs_matrix.size() - 2)
            combine_matrix[i] = tfs_matrix[i].matrix();*/
        else
            combine_matrix[i] = combine_matrix[i - 1] * tfs_matrix[i].matrix();
    }

    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType mergedCloud(new pcl::PCLPointCloud2);
    CloudWithoutType baseCloud(new pcl::PCLPointCloud2);
    std::string inputbaseCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[0].second; // first
    iospace::loadPlyFile(inputbaseCloud, baseCloud);
    Eigen::Matrix4f base_matrix = Eigen::Matrix4f::Identity();
    mergedCloud = tool::TransFormationOfCloud(baseCloud, base_matrix);
    int num_view_pair = m_SimulatedViewPairs.size(); // m_SimulatedViewPairs.size() -1 ;
    for (int pa = 0; pa < m_SimulatedViewPairs.size(); pa++) //m_SimulatedViewPairs.size()
    {
        std::cout << "Evaluation pair:" << pa + 1 << "/" << num_view_pair << std::endl;
        std::string source_view = m_SimulatedViewPairs[pa].second;
        std::string  target_view = m_SimulatedViewPairs[pa].first;

        if (source_view == "" || target_view == "")
        {
            throw runtime_error("ERROR: Unable to read view pair for analysis\n");
        }
        // load input data 
        std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].second;
        std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].first;

        iospace::loadPlyFile(inputTargetCloud, sourceCloud);

       
        iospace::loadPlyFile(inputSourceCloud, targetCloud);
        std::cout << "Loaded Files" << "source:" << source_view << "," << "target:" << target_view << std::endl;
        CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
        TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, combine_matrix[pa]);
        mergedCloud = iospace::MergeCloud(mergedCloud, TransformedSourceCloud);
       
    }
    string mergedFileName = this->m_DataSetPath + "/" +"banana" + "_merged_sicp.ply";
    pcl::PLYWriter writer;
    writer.writeBinary(mergedFileName, *mergedCloud);
   // iospace::writePlyFile(mergedFileName, mergedCloud);
}

void Baseline::BaseLine::FilterCloud() 
{
    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType filtered_sourceCloud(new pcl::PCLPointCloud2);
    for (int pa = 0; pa < m_SimulatedViewPairs.size(); pa++)
    {
        std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].first;  //  flip to this->m_DataSetPath after sicp run
        iospace::loadPlyFile(inputSourceCloud, sourceCloud);
        filtered_sourceCloud = FilterPlaneFromPointCloud(sourceCloud);
        iospace::writePlyFile(inputSourceCloud, filtered_sourceCloud);
       
    }
}
void Baseline::BaseLine::RegisterMultipleDataPairwise()
{
    double stop_criteria = 1e-6f;
    GeneratePairWiseViewPair(m_DataSetPath, m_FileExtension);
    m_OutPutTransformations.resize(m_SimulatedViewPairs.size());
    // filter cloud if required
   // FilterCloud();
        
    // allocate respective clouds
    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType filtered_sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType filtered_targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);

    std::string source_view = "";
    std::string  target_view = "";
    std::string inputTargetCloud = "";
    ////////diagonal length computation end////////////////////////////
    for (int pa = 0; pa <m_SimulatedViewPairs.size(); pa++) // m_SimulatedViewPairs.size() - 1
    {

        std::cout << "Evaluation pair:" << pa + 1 << "/" << m_SimulatedViewPairs.size() << std::endl;
        
        // load input data 
        std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].second;
        std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].first;
        iospace::loadPlyFile(inputTargetCloud, sourceCloud);
        iospace::loadPlyFile(inputSourceCloud, targetCloud);
        std::cout << "Loaded Files" << "source:" << this->m_SimulatedViewPairs[pa].second << "," << "target:" << this->m_SimulatedViewPairs[pa].first << std::endl;


        ///////method for inverse calibration method////////////////
        if (inverse_calibration_method == true)
        {
            LoadPCTFiles(pa);
        }

        double smoothing_time = 0.0f;
        error_log("/////////Computation Begin///////////////////\n");
        // instantiate normal sampling
        cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
        nml_sampling.PrepareSamplingData(sourceCloud);
        CpuTimeProfiler cpuTime;
        nml_sampling.SamplePointCloud(sampledSourceCloud);
       // this->m_samplingTime[pa] = cpuTime.GetElapsedSecs();
        Eigen::Matrix4f output_transformation = Eigen::Matrix4f::Identity();
       /* string sampled_ply = m_BaseFolderPath + "/sampled.ply";
        pcl::io::savePLYFile(sampled_ply, *sampledSourceCloud);*/
        if (m_MethodName == "fgr")
            PairWiseRegistration(sourceCloud, targetCloud, inputTargetCloud, pa, output_transformation,
                smoothing_time, stop_criteria); // change for others
        else
            PairWiseRegistration(sampledSourceCloud, targetCloud, inputTargetCloud, pa, output_transformation,
                smoothing_time, stop_criteria,initialPoses[pa]); // change for others



        //////////////GetmergedCloud/////////////////////

        TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, output_transformation);
        this->m_OutPutTransformations[pa] = output_transformation;
         string mergedFileName = m_BaseFolderPath + "/" + source_view + "_" + target_view + "_merged.ply";
       // iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

       
    }
    // Collect Data for each pair of simulated scans
    SaveTransformationMatrices(m_transformFileName);
}

CloudWithoutType Baseline::BaseLine::FilterPlaneFromPointCloud(CloudWithoutType &inputCloud)
{
    CloudWithNormalPtr CloudThree(new pcl::PointCloud <PointNormalType>);
    CloudWithNormalPtr Cloud_filtered(new pcl::PointCloud <PointNormalType>);
    pcl::fromPCLPointCloud2(*inputCloud, *CloudThree);
//    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//    // Create the segmentation object
//    pcl::SACSegmentation<PointType> seg;
//    // Optional
//    seg.setOptimizeCoefficients(true);
//    seg.setMaxIterations(50);
//    // Mandatory
//    seg.setModelType(pcl::SACMODEL_PLANE);
//    seg.setMethodType(pcl::SAC_RANSAC);
//    seg.setDistanceThreshold(0.01);
//    //Eigen::Vector3f axis = Eigen::Vector3f(0.0, 1.0, 0.0); //y axis
//    //seg.setAxis(axis);
//    seg.setInputCloud(CloudThree);
//    seg.segment(*inliers, *coefficients);
//
//    // Create the filtering object
//    CloudWithNormalPtr CloudNormal(new pcl::PointCloud <PointNormalType>);
//    pcl::fromPCLPointCloud2(*inputCloud, *CloudNormal);
//    CloudWithNormalPtr PointNormals(new pcl::PointCloud<PointNormalType>);
//    size_t cloud_pts = CloudThree->points.size() - inliers->indices.size();
//    PointNormals->width = cloud_pts;
//    PointNormals->height = 1;
//    PointNormals->points.resize(PointNormals->width * PointNormals->height);
//  
//    
//    int i;
//    int curr_index = 0;
////#pragma omp parallel for
//    for (i = 0; i < CloudThree->points.size(); i++)
//    {
//        if (std::find(inliers->indices.begin(), inliers->indices.end(), i) == inliers->indices.end())
//        {
//            PointNormals->points[curr_index].getVector3fMap() = CloudNormal->points[i].getVector3fMap();
//            PointNormals->points[curr_index].getNormalVector3fMap() = CloudNormal->points[i].getNormalVector3fMap();
//            curr_index++;
//        }
//       /* else
//        {
//            float n =  std::numeric_limits<float>::quiet_NaN();
//            PointNormals->points[i].getVector3fMap() = Eigen::Vector3f(n, n, n);
//           
//        }*/
//    }
    std::vector<int> indices;
    CloudWithNormalPtr new_PointNormals(new pcl::PointCloud<PointNormalType>);
    pcl::PassThrough<PointNormalType> pass_thru_filter;
    pass_thru_filter.setFilterFieldName("x");
    pass_thru_filter.setFilterLimitsNegative(true);
    pass_thru_filter.setFilterLimits(-200.0, -55.0);
    pass_thru_filter.setInputCloud(CloudThree);
    //pass_thru_filter.setKeepOrganized(false);
    pass_thru_filter.filter(*Cloud_filtered);
   
   /* CloudWithNormalPtr new_PointNormals(new pcl::PointCloud<PointNormalType>);
    pcl::removeNaNFromPointCloud(*PointNormals, *new_PointNormals, indices);*/

   /* pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(CloudThree);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*Cloud_filtered);*/
    CloudWithoutType newCloud(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*Cloud_filtered, *newCloud);
    return newCloud;
}
void Baseline::BaseLine::EvaluateAllViewPairForSimulatedScans()
{
    GenerateSimulatedScanViewPair(m_DataSetPath, m_MeshDataPath, m_FileExtension);
    m_OutPutTransformations.resize(m_SimulatedViewPairs.size());
    m_vPairwiseCpuTimes.resize(m_SimulatedViewPairs.size(), 0.0);
    RMSEs.resize(m_SimulatedViewPairs.size(), 0.0);
    m_samplingTime.resize(m_SimulatedViewPairs.size(), 0.0);
    m_smoothingTime.resize(m_SimulatedViewPairs.size(), 0.0);
    m_registration_time.resize(m_SimulatedViewPairs.size(), 0.0);
    m_corresp_time.resize(m_SimulatedViewPairs.size(), 0.0);
    optim_time.resize(m_SimulatedViewPairs.size(), 0.0);
    // allocate respective clouds
    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType inputTargetGroundTruthCloud(new pcl::PCLPointCloud2);
    CloudWithoutType inputSourceGroundTruthCloud(new pcl::PCLPointCloud2);

    if (inverse_calibration_method == true)
    {
        GeneratePctViewPairs(m_DataSetPath, m_MeshDataPath, m_pctExtension);
    }
    ///////////diagonal length computation begin////////////////////
    CloudWithoutType SourceGroundTruthCloud(new pcl::PCLPointCloud2);
    std::string inputSourceGroundTruthFile = this->m_MeshDataPath + "/" + this->m_SimulatedGroundTruthViewPairs[0].second;
    iospace::loadPlyFile(inputSourceGroundTruthFile, SourceGroundTruthCloud);
    Eigen::Vector3f min_pt, max_pt;
    CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
    pcl::fromPCLPointCloud2(*SourceGroundTruthCloud, *tar_new);
    double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
    std::cout << "Diagonal Length Computed" << diagonalLength << std::endl;
    m_diagonalLength = diagonalLength;

////////diagonal length computation end////////////////////////////
    for (int pa = 0; pa < m_SimulatedViewPairs.size(); pa++) //m_SimulatedViewPairs.size()
    {
       
        std::cout << "Evaluation pair:" << pa + 1 << "/" << m_SimulatedViewPairs.size() << std::endl;
        std::string source_view = m_SimulatedViewPairs[pa].second;
        std::string  target_view = m_SimulatedViewPairs[pa].first;

        std::string inputSourceGroundTruthFile = this->m_MeshDataPath + "/" + this->m_SimulatedGroundTruthViewPairs[pa].second;
        std::string inputTargetGroundTruthFile = this->m_MeshDataPath + "/" + this->m_SimulatedGroundTruthViewPairs[pa].first;
        iospace::loadPlyFile(inputSourceGroundTruthFile, inputSourceGroundTruthCloud);
        iospace::loadPlyFile(inputTargetGroundTruthFile, inputTargetGroundTruthCloud);
        m_sourceGroundTruth = inputSourceGroundTruthCloud;
        m_targetGroundTruth = inputTargetGroundTruthCloud;
        if (source_view == "" || target_view == "")
        {
            throw runtime_error("ERROR: Unable to read view pair for analysis\n");
        }
       
       // load input data 
        std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].second;
        iospace::loadPlyFile(inputSourceCloud, sourceCloud);

        std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_SimulatedViewPairs[pa].first;
        iospace::loadPlyFile(inputTargetCloud, targetCloud);
        std::cout << "Loaded Files" << "source:" << source_view << "," << "target:" << target_view << std::endl;

      
        ///////method for inverse calibration method////////////////
        if (inverse_calibration_method == true)
        {
            LoadPCTFiles(pa);
        }

        //std::cout << "diagonalLength:" << diagonalLength << std::endl;
        double stop_criteria = ComputeStopCriteria(offset, diagonalLength, this->Error_StdDev);

        //////////////////feature angle///////////////////////
       /* featureSource = metric::ComputeFeatureAngleForACloud(sourceCloud, 11);
        featureTarget = metric::ComputeFeatureAngleForACloud(targetCloud, 11);  */
         /////////////////////////////////////////////
      
        // Compute Prinicipal Frame for Filtering
        int polynomial_degree = 3.0;
        int mls_iteration = 20;
        bool with_filter = false;
        CWendlandWF<float> wendLandWeightF;
        // computer local reference frame for source cloud
        cMLSCorrespondence mlsEstimateA(sourceCloud, wendLandWeightF, m_searchRadius, polynomial_degree, mls_iteration);
        CpuTimeProfiler cpuTimeB;
      sourceCloud =  mlsEstimateA.ComputeMLSSurface();
      double  mlsTime = cpuTimeB.GetElapsedSecs();
      src_principal_frame = mlsEstimateA.GetPrincipalFrame();
      source_eig_value = mlsEstimateA.GetEigenValue();
       // sourceCloud = mlsEstimateA.GetInputCloud();  // update source cloud to exclude any unwanted spuriuos points with no prinicipal frame
        CreateIndicesVsPointMap(sourceCloud);

        error_log("/////////Computation Begin///////////////////\n");
        // instantiate normal sampling
        cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
        nml_sampling.PrepareSamplingData(sourceCloud);
        CpuTimeProfiler cpuTime;
        nml_sampling.SamplePointCloud(sampledSourceCloud);
        this->m_samplingTime[pa] = cpuTime.GetElapsedSecs();
        Eigen::Matrix4f output_transformation = Eigen::Matrix4f::Identity();
        string sampled_ply = m_BaseFolderPath + "/sampled.ply";
        pcl::io::savePLYFile(sampled_ply, *sampledSourceCloud);
      this->m_registration_time[pa] = PairWiseRegistration(sampledSourceCloud, targetCloud, inputTargetCloud, pa, output_transformation,
            this->m_smoothingTime[pa], stop_criteria);


        //////////////GetmergedCloud/////////////////////
        
        TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, output_transformation);
        CloudWithoutType mergedCloud;
        std::string Source_Gt_Base_Name = this->m_SimulatedViewPairs[pa].second.substr(0, this->m_SimulatedViewPairs[pa].second.find("."));
        std::string Target_Gt_Base_name = this->m_SimulatedViewPairs[pa].first.substr(0, this->m_SimulatedViewPairs[pa].first.find("."));
        string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
       /* string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
        iospace::writePlyFile(mergedFileName, TransformedSourceCloud);*/
        this->m_OutPutTransformations[pa] = output_transformation;

        // RMSE computation
        CloudWithoutType tranformed_groundTruth_source = tool::TransFormationOfCloud(inputSourceGroundTruthCloud, output_transformation);
        double rmse = static_cast<double>(metric::ComputeRMSE(tranformed_groundTruth_source, inputTargetGroundTruthCloud, 1.0));
        RMSEs[pa] = rmse;
    }
    // Collect Data for each pair of simulated scans
    SaveTransformationMatrices(m_transformFileName);
    WriteProcessingTime(m_processingTimeFileName);  // writes processing each separate task of algorithm
    string DataSetName = SplitFolderNameFromPath(m_DataSetPath);
    string fileName = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + m_MethodName + "_" + DataSetName + "_icp_component_time.txt";
    WriteCorrespondenceAndOptimizationTime(fileName);  // writes correspondence and optimization time for icp all the views together
    SaveRMSE(m_RmseFileName);
}
string Baseline::BaseLine::SplitFolderNameFromPath(string &FolderWithPath)
{
    std::size_t found = FolderWithPath.find_last_of("/\\");
    string FolderName  =  FolderWithPath.substr(found + 1);
    return FolderName;
}
void Baseline::BaseLine::RegisterMultipleData()
{
    int numFolder = m_DataFolderList.size();
    int numGroundtruthFolder = m_groundTruthFileList.size();
    if (numFolder == 0)
    {
        throw runtime_error("ERROR: No Data to Evaluate\n");
    }
    for (int ItrFolder = 0; ItrFolder < numFolder; ItrFolder++)
    {
        string FolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder];
        std::string transform_fileName = m_DataFolderList[ItrFolder] + "_pairwise_transform_"
            + m_MethodName + ".txt";
        std::string view_pair = m_DataFolderList[ItrFolder] + "_registration_view_pair_Baseline_"
            + m_MethodName + ".txt";
        SetDataPath(FolderPath);
        setOutPutTransformationFileName(transform_fileName);
        /*if (m_MethodName != "fgr")
            initialPoses = tool::ReadTransformationMatrix(FolderPath + "/" + m_DataFolderList[ItrFolder] + "_pairwise_transform_fgr.txt");*/
       RegisterMultipleDataPairwise();
      // if (m_MethodName == "fgr")
           TransformSuccessiveViewPair(transform_fileName);
    }
}
void Baseline::BaseLine::PrepareEvaluationOFSimulatedScans()
{
    int numFolder = m_DataFolderList.size();
    int numGroundtruthFolder = m_groundTruthFileList.size();
    if (numFolder == 0 && numGroundtruthFolder!=numFolder)
    {
        throw runtime_error("ERROR: No Data to Evaluate\n");
    }
    for (int ItrFolder = 0; ItrFolder < numFolder; ItrFolder++)
    {
        string FolderPath = m_BaseFolderPath + "/" + m_DataFolderList[ItrFolder];
        string GroundTruthFolderpath = m_BaseFolderPath + "/" + m_groundTruthFileList[ItrFolder];
        std::string transform_fileName = m_DataFolderList[ItrFolder] + "_pairwise_transform_20_new_samples_"
            + m_MethodName + ".txt";
        std::string view_pair = m_DataFolderList[ItrFolder] + "_registration_view_pair_Baseline_"
            + m_MethodName + ".txt";
        std::string rmse_filename = m_DataFolderList[ItrFolder] + "_rmse_20_new_samples_" + m_MethodName + ".txt";
        std::string output_fileName = m_DataFolderList[ItrFolder] + "_performance_metric_Baseline_"
            + m_MethodName + ".txt";
        std::string MergeLocation = "Baseline_20_new_samples_" + m_MethodName;
        SetDataPath(FolderPath); // set evaluationfolder path
        MeshDataPath(GroundTruthFolderpath);  //set groundtruthfolder path for rmse computation
        SetOutPutDirectoryPath(MergeLocation);
        setOutPutTransformationFileName(transform_fileName);
        string processingtime = FolderPath + "/" + m_DataFolderList[ItrFolder] + "_processing_time_Baseline_20_new_samples_"
            + m_MethodName + ".txt";  //FolderPath
       /* string sampled_fileName = FolderPath + "/" + MergeLocation + "/" + m_DataFolderList[ItrFolder] + "_processing_time_sampled_20_new.txt";*/
        setProcessingTimeFileName(processingtime);
        m_RmseFileName = rmse_filename;
        if (inverse_calibration_method == true)
        {
            string fileName = m_BaseFolderPath + "/" + "ProjectionFileName.txt";
            LoadProjectionMatrix(fileName, ItrFolder);
        }
        EvaluateAllViewPairForSimulatedScans();

       // SaveTransformationMatrices(transform_fileName);
        Reset();
       
    }
}
bool  Baseline::BaseLine::GeneratePctViewPairs(const std::string &dirEvalName, const std::string &dirGroundTruthName, const std::string &fileExt)
{
    if (fileExt == "pct")
    {
        std::vector<std::string>fileNames;
        iospace::FindFilesEndWith(dirEvalName, fileExt, fileNames, true);
        std::vector<std::pair<std::string, std::string>> view_pair_list;
        int numViewpair = fileNames.size();
        int ViewPair = m_SimulatedViewPairs.size();
        if (numViewpair != ViewPair)
        {
            throw runtime_error("ERROR: evaluation and groundtruth view pair does not match");
        }
        view_pair_list.reserve(numViewpair);

        for (int i = 0; i < fileNames.size(); i++)
        {
            int j = i + 1;
            if (j != fileNames.size())
            {
                if (fileNames[j].compare(fileNames[i]) != 0)
                {
                    std::pair<std::string, std::string> view_pair(fileNames[i], fileNames[j]);
                    view_pair_list.push_back(view_pair);

                }

            }
        }
        std::pair<std::string, std::string> view_pair(fileNames[fileNames.size() - 1], fileNames[0]);
        view_pair_list.push_back(view_pair);
        m_SimulatedPctViews = view_pair_list;
        return true;
    }
    else
    {
        return false;
    }
  
}

std::vector<UVData<float>> Baseline::BaseLine::LoadPCTFiles(int fileCounter)
{
    // load target pct file
    std::vector <UVData<float>>pixels;
    std::vector<PointType>point_cloud;
    std::vector<float>intensity;
    std::string targetpctcloud = this->m_DataSetPath + "/" + m_SimulatedPctViews[fileCounter].first;
    pct::LoadPCTFIle(targetpctcloud, pixels, intensity, point_cloud);
    m_TargetPixels = pixels;
    return pixels;
}

void Baseline::BaseLine:: LoadPCDAndSavePly(const std::string &dirEvalName, const std::string &fileExt)
{
    std::vector<std::string>fileNames, groundtruthFilename;
    iospace::FindFilesEndWith(dirEvalName, fileExt, fileNames, true);
    for (int i = 0; i < fileNames.size(); i++)
    {

    }
}
void Baseline::BaseLine::GeneratePairWiseViewPair(const std::string &dirEvalName, const std::string &fileExt)
{
    std::vector<std::string>fileNames, groundtruthFilename;
    iospace::FindFilesEndWith(dirEvalName, fileExt, fileNames, true);
    std::vector<std::pair<std::string, std::string>> view_pair_list;
    int numViewpair = (fileNames.size());
    view_pair_list.reserve(numViewpair);
 
    for (int i = 0; i < fileNames.size(); i++)
    {
        int j = i + 1;
        if (j != fileNames.size())
        {
            if (fileNames[j].compare(fileNames[i]) != 0)
            {
                std::pair<std::string, std::string> view_pair(fileNames[i], fileNames[j]);
                view_pair_list.push_back(view_pair);
                
            }

        }
    }
   /* std::pair<std::string, std::string> view_pair (fileNames[0], fileNames[fileNames.size() - 1]);
    view_pair_list.push_back(view_pair);*/
    m_SimulatedViewPairs = view_pair_list;
}
void Baseline::BaseLine::GenerateSimulatedScanViewPair(const std::string &dirEvalName, const std::string &dirGroundTruthName, const std::string &fileExt)
{
    std::vector<std::string>fileNames, groundtruthFilename;
    iospace::FindFilesEndWith(dirEvalName, fileExt, fileNames, true);
    std::vector<std::pair<std::string, std::string>> view_pair_list, groundtruth_view_pair;
    iospace::FindFilesEndWith(dirGroundTruthName, fileExt, groundtruthFilename, true);
    int numViewpair = (fileNames.size());
    int groundtruthViewPair = groundtruthFilename.size();
    if (numViewpair != groundtruthViewPair)
    {
        throw runtime_error("ERROR: evaluation and groundtruth view pair does not match");
    }
    view_pair_list.reserve(numViewpair);
    groundtruth_view_pair.reserve(groundtruthViewPair);

    for (int i = 0; i < fileNames.size(); i++)
    {
        int j = i + 1;
        if(j!= fileNames.size())
        {
            if (fileNames[j].compare(fileNames[i]) != 0)
            {
                std::pair<std::string, std::string> view_pair(fileNames[i], fileNames[j]);
                view_pair_list.push_back(view_pair);
                if (0 != groundtruthViewPair)
                {
                    std::pair<std::string, std::string> gt_view_pair(groundtruthFilename[i], groundtruthFilename[j]);
                    groundtruth_view_pair.push_back(gt_view_pair);
                }
            }
           
        }
    }
    std::pair<std::string, std::string> gt_view_pair(groundtruthFilename[groundtruthFilename.size() - 1], groundtruthFilename[0]);
    groundtruth_view_pair.push_back(gt_view_pair);
    std::pair<std::string, std::string> view_pair(fileNames[fileNames.size() -1 ], fileNames[0]);
    view_pair_list.push_back(view_pair);
    m_SimulatedViewPairs = view_pair_list;
    m_SimulatedGroundTruthViewPairs = groundtruth_view_pair;

}
void Baseline::BaseLine::SetParamters(double searchRadius, int numSamples)
{
    m_searchRadius = searchRadius;
    m_SampleSize = numSamples;
}

void Baseline::BaseLine::ExtractStandardDeviation(double &Value, string fileName)
{
    string std_dev_FileName = m_DataSetPath + "/" + m_MethodName + "_" + fileName ;
    bool fileExists = iospace::ExistsFile(std_dev_FileName);

    if (fileExists && (std_dev_FileName != ""))
    {
        iospace::LoadSingleValue<double>(std_dev_FileName, Value);
        Error_StdDev = Value;
    }
    else
    {
        iospace :: SaveSingleValue<double>(std_dev_FileName, Value);
    }
}

double Baseline::BaseLine::ComputeStopCriteria(double OffSet, float unit, double std_dev)
{
    string std_dev_FileName = m_DataSetPath + "/" + m_MethodName + "_" + "standard_deviation_error.txt";
    if (std_dev == 1e-7f && !iospace::ExistsFile(std_dev_FileName))
    {
        StopCriteria = 1e-7f;
        return StopCriteria;

    }
    ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");
    double eta = (std_dev / unit) * OffSet;
    StopCriteria = std_dev;// *unit; //std_dev *unit
    std::cout << "StopCriteria as usual:" << StopCriteria << std::endl;
    std::cout << "stopCriteria scaled:" << std_dev * unit << std::endl;
    return StopCriteria; // 2.0e-4 * unit;
}

void Baseline::BaseLine::ReadInputCloudForEvaluation(string &inputSourceFile, string &inputTargetFile, int index,
    CloudWithoutType &CloudSource, CloudWithoutType &CloudTarget)
{
  
    if (inputSourceFile.compare(m_ViewPairs.at(index).second) != 0)
    {
        inputSourceFile = m_ViewPairs.at(index).second;
        std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[index].second;  //  flip to this->m_DataSetPath after sicp run
        iospace::loadPlyFile(inputSourceCloud, CloudSource);
    }
    if (inputTargetFile.compare(m_ViewPairs.at(index).first) != 0 )
    {
        inputTargetFile = m_ViewPairs.at(index).first;
        std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[index].first;  // flip  to this->m_DataSetPath
        iospace::loadPlyFile(inputTargetCloud, CloudTarget);
    }

   
}
double Baseline::BaseLine::GetTotalCPUTime()
{
    return m_TotalTime;
}

void Baseline::BaseLine::ReadFolderNameFromList(string &FileName)
{
    string NameAndFolder = m_BaseFolderPath + "/" + FileName;
    std::string line;
    std::ifstream myfile(NameAndFolder.c_str());
    m_DataFolderList.reserve(100);
    unsigned numberOfLines = 0;
    if (myfile.is_open())
    {
        while (std::getline(myfile, line))
        {
            m_DataFolderList.push_back(line);
            numberOfLines++;
        }
        std::cout << "Number of DataFolder: " << numberOfLines << std::endl;
    }
    myfile.close();

}
void Baseline::BaseLine::SetGroundTruthFolderNames(const string &DirName)
{
    string NameAndFolder = m_BaseFolderPath + "/" + DirName;
    std::string line;
    std::ifstream myfile(NameAndFolder.c_str());
    m_groundTruthFileList.reserve(100);
    unsigned numberOfLines = 0;
    if (myfile.is_open())
    {
        while (std::getline(myfile, line))
        {
            m_groundTruthFileList.push_back(line);
            numberOfLines++;
        }
        std::cout << "Number of GroundTruthFolder: " << numberOfLines << std::endl;
    }
    myfile.close();
}

void Baseline::BaseLine::WriteProcessingTime(string &FileName)
{
    FILE	*g_pLogFile = fopen(FileName.c_str(), "wb");
    if (NULL == g_pLogFile)
    {
        abort();
    }
    fprintf(g_pLogFile, "\samplingTime\tsmoothingTime\tregistration_time\n");
    
        for (int i = 0; i < m_registration_time.size(); i++)
        {
            fprintf(g_pLogFile, "%.4f\t%.4f\t%.4f\n", m_samplingTime[i], m_smoothingTime[i], m_registration_time[i]);
        }
        if (NULL != g_pLogFile)
        {
            fclose(g_pLogFile);
            g_pLogFile = NULL;
        }
 
}
void Baseline::BaseLine::WriteCorrespondenceAndOptimizationTime(string &FileName)
{
    FILE	*g_pLogFile = fopen(FileName.c_str(), "wb");
    if (NULL == g_pLogFile)
    {
        abort();
    }
    fprintf(g_pLogFile, "\corresp_time\toptim_time\tregistration_time\n");

    for (int i = 0; i < m_registration_time.size(); i++)
    {
        fprintf(g_pLogFile, "%.4f\t%.4f\t%.4f\n", m_corresp_time[i], optim_time[i], m_registration_time[i]);
    }
    if (NULL != g_pLogFile)
    {
        fclose(g_pLogFile);
        g_pLogFile = NULL;
    }
}
void Baseline::BaseLine::ReadRegistrationTime(string &fileName, std::vector<double>&correstime, std::vector<double>&optim_time, std::vector<double>&regis_time)
{
    correstime.clear();
    correstime.shrink_to_fit();
    optim_time.clear();
    optim_time.shrink_to_fit();
    regis_time.clear();
    regis_time.shrink_to_fit();
    unsigned noOfLines = pct::readLines(fileName);
    int fileRead = 0;
    FILE *pFile;
    pFile = fopen(fileName.c_str(), "rb");
    if (NULL == pFile)
    {
        std::cout << "Failed to read data file " << std::endl;
        throw std::runtime_error(" unable to open file\n");
    }
    char szParam1[50], szParam2[50], szParam3[50];
    fscanf(pFile, "%s %s %s", szParam1, szParam2, szParam3);
    for (int i = 1; i < noOfLines; i++)
    {
        fscanf(pFile, "%s %s %s\n", szParam1, szParam2, szParam3);
        double n_time = atof(szParam1);
        double s_time = atof(szParam2);
        double r_time = atof(szParam3);
        correstime.push_back(n_time);
        optim_time.push_back(s_time);
        regis_time.push_back(r_time);
    }
    fclose(pFile);
}
void Baseline::BaseLine::ReadRegistrationTimeAndRMSEForAllDataset(string & sourceFileA, string &sourceFileB,
    std::vector<double>&rmse_avg, std::vector<double>&time_avg)
{
    /*string inputfiledirectory = m_DataSetPath + "/" + "SampleVsTimeResult";
    string inputfileListA = inputfiledirectory + "/" + sourceFileA;
    string inputfileListB = inputfiledirectory + "/" + sourceFileB;

    std::string lineA;
    std::ifstream myfileA(inputfileListA.c_str());
    std::string lineB;
    std::ifstream myfileB(inputfileListB.c_str());
    if (myfileA.is_open())
    {
        while (std::getline(myfileA, lineA))
        {

        }
    }
    std::string lineB;
    std::ifstream myfileB(inputfileListB.c_str());*/
}
Eigen::Matrix4f Baseline::BaseLine::LoadProjectionMatrix(string &fileNameA,int folderIndex)
{
    std::ifstream myfile(fileNameA.c_str());
    std::vector<string>projection_matrix_list;
    projection_matrix_list.reserve(4);
    if (myfile.is_open())
    {
        while (std::getline(myfile, fileNameA))
        {
            projection_matrix_list.push_back(fileNameA);
        }
    }
    myfile.close();
std:string PjFileName = m_BaseFolderPath + "/" + projection_matrix_list[folderIndex];
    m_projectionMatrix = tool::LoadProjectionmatrix(PjFileName);
    projection_matrix_list.clear();
    projection_matrix_list.shrink_to_fit();
    return m_projectionMatrix;
}
void Baseline::BaseLine::SetInverseCalibrationMethod(bool option)
{
    inverse_calibration_method = option;
}

void Baseline::BaseLine::CreateIndicesVsPointMap(CloudWithoutType & PointCloud)
{
    CloudWithNormalPtr  temp_cloud(new pcl::PointCloud<PointNormalType>);
    pcl::fromPCLPointCloud2(*PointCloud, *temp_cloud);
    Principal_frame_Index_source_map.clear();
    for (int i = 0; i < temp_cloud->points.size(); i++)
    {
        PointDataType pt(temp_cloud->points[i].x, temp_cloud->points[i].y, temp_cloud->points[i].z,
            temp_cloud->points[i].normal_x, temp_cloud->points[i].normal_y, temp_cloud->points[i].normal_z);
        Principal_frame_Index_source_map.insert(std::pair<PointDataType, int>(pt, i));
    }
}

void Baseline::BaseLine_WithTargetSmoothing::ComputeNormalSmoothingUsingMLSProjection(std::vector<std::pair<std::string, std::string>> &views,
    string &inputFileName, int index, CloudWithoutType &inputCloud, cMLSCorrespondence &mlsCorres, CloudWithoutType &SmoothedNormalCloud)
{
    if (inputFileName.compare(views.at(index).first) != 0)
    {
        inputFileName = views.at(index).first;
        CloudWithoutType mls_targetCloud = mlsCorres.ComputeMLSSurface();
        if (true == m_WithNormalSmoothing)  // only uses normal of smooth cloud
            SmoothedNormalCloud = tool::CopytNormalWithoutPoint(inputCloud, mls_targetCloud);
        else
            SmoothedNormalCloud = mls_targetCloud;
       // SmoothedNormalCloud = normal_smoothed_targetCloud;
    }
}
void Baseline::BaseLine_WithTargetSmoothing::Evaluate()
{
    int size = this->m_ViewPairs.size();
    string sourceInput = "";
    string targetInput = "";
    string smoothTargetFile = "";
    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType normal_smoothed_targetCloud(new pcl::PCLPointCloud2);
    for (int i = 0; i < size; i++)
    {
        std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
        // ReadViewpair for Evaluation 
        
        ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
        //// load source ply file
       
        //std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[i].second;
        //iospace::loadPlyFile(inputSourceCloud, sourceCloud);

        Eigen::Vector3f min_pt, max_pt;
        CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
        pcl::fromPCLPointCloud2(*sourceCloud, *tar_new);
        double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
      /*  double eta = (this->Error_StdDev / diagonalLength) * 1.5;
        double stop_criteria = eta * diagonalLength;*/
        double stop_criteria = ComputeStopCriteria(stop_criteria_multiplier, diagonalLength, this->Error_StdDev);


        ////read target cloud
       
        //std::string inputtargetCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[i].first;
        //iospace::loadPlyFile(inputtargetCloud, targetCloud);

        // load target pct file
        std::vector <UVData<float>>pixels;
        std::vector<PointType>point_cloud;
        std::vector<float>intensity;

        //// std::string targetPCTCloud = "C:/WorkSpace/Regis_3D/Regis_3D/bimba_0.pct";
        //pct::LoadPCTFIle(parser::target_pctfile_, pixels, intensity, point_cloud);

        // load projection matrix
        std::string projectionFile = "C:/WorkSpace/Regis_3D/Regis_3D/bimba_ProjMat_fov60.txt";
        // Eigen::Matrix4f proj_mat = tool::LoadProjectionmatrix(projectionFile);


        error_log("/////////Computation Begin///////////////////\n");
        // instantiate normal sampling

        size_t sampling_size = sourceCloud->height * sourceCloud->width;
        CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
        cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
        nml_sampling.PrepareSamplingData(sourceCloud);
        CpuTimeProfiler cpuTime;
        nml_sampling.SamplePointCloud(sampledSourceCloud);
       this->m_samplingTime[i] = cpuTime.GetElapsedSecs();

        //Instantiating weight function
        CWendlandWF<float> wendLandWeightF;

        //// Compute Prinicipal Frame for Filtering
        //cMLSCorrespondence mlsEstimateA(sampledSourceCloud, wendLandWeightF, this->m_searchRadius, 3.0f, 20);
        //mlsEstimateA.ComputeMLSSurface();
        //std::vector<Eigen::Matrix3f>source_principal_frame = mlsEstimateA.GetPrincipalFrame();
        //std::vector<Eigen::Vector3f>source_eigen_value = mlsEstimateA.GetEigenValue();

         // Smooth Normals using MLS projection
        cMLSCorrespondence mlsEstimate_target(targetCloud, wendLandWeightF, this->m_searchRadius, 3.0f, 20);
        /*CloudWithoutType mls_targetCloud = mlsEstimate_target.ComputeMLSSurface();
        CloudWithoutType normal_smoothed_targetCloud = tool::CopytNormalWithoutPoint(targetCloud, mls_targetCloud);*/
        CpuTimeProfiler cpuTimeB;
        ComputeNormalSmoothingUsingMLSProjection(this->m_ViewPairs, smoothTargetFile, i, targetCloud,
            mlsEstimate_target, normal_smoothed_targetCloud);
        this->m_smoothingTime[i] = cpuTimeB.GetElapsedSecs();

        cKDTreeSearch search(true, 12.0f, false);
        cCorrespondenceKdtree cKdTree(search);

        cTransformEstimationSparseICP regis_3d_kdtree;
       
        // Registration Call
        Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
        initTransformation = InitialPose_Eth[i];// initialPoses[i]; ;  // 1) InitialPose_Eth = eth data
                                                                       //   2) initialPoses = LRF
          float norm = 0.5f;
        int numIcpIteration = 100;
        int searchStrategy = 0;
        Eigen::Matrix4f finalMatrix;
        finalMatrix.setIdentity();
        Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
        bool fallback = false;

        regis_3d_kdtree.PrepareSample(sampledSourceCloud, normal_smoothed_targetCloud);
        regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
            stop_criteria, fallback, diagonalLength);
        CpuTimeProfiler cpuTimeC;
        std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&cKdTree, finalMatrix);
        this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
        double corres_time, optim_time;
        regis_3d_kdtree.GetIcpTimings(corres_time, optim_time);
        this->m_corresp_time[i] = corres_time;
        this->optim_time[i] = optim_time;

        std::vector<Eigen::Matrix4f> output_matrix(1);
        this->m_OutPutTransformations[i] = finalMatrix;
        double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
        ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

        ////////////GetmergedCloud/////////////////////
        CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
        TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
        CloudWithoutType mergedCloud;

        std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
        std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
        string newDirectory = m_DataSetPath + "/" +  this->m_OutputMergedDir;
        string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
       // iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

        error_log("/////////////////Computation End///////////////\n");
        EndDataLog();

        /////////RMSE Computation begins//
        // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
        //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
        //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
        if (false == eth_evaluation)
        {
            /////////RMSE Computation begins//
            // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
            //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
            //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
            double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, this->m_ViewPairs[i].second,
                this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
            std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
            RMSEs[i] = rmse_value;
        }
        else
        {
            Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], finalMatrix);
            CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
            CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
            RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
        }
    }
}
 double Baseline::BaseLine_WithTargetSmoothing::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
     string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation, double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
{
    // static int i = 0;
     double regis_time = 0.0;
    //Instantiating weight function
    CWendlandWF<float> wendLandWeightF;
  
    // Smooth Normals using MLS projection
    cMLSCorrespondence mlsEstimate_target(targetCloud, wendLandWeightF, this->m_searchRadius, 3.0f, 20);
   
    CloudWithoutType normal_smoothed_targetCloud(new pcl::PCLPointCloud2);
    CpuTimeProfiler cpuTimeB;
    ComputeNormalSmoothingUsingMLSProjection(this->m_SimulatedViewPairs, CloudTarget, index, targetCloud,
        mlsEstimate_target, normal_smoothed_targetCloud);
    mlsTime = cpuTimeB.GetElapsedSecs();
    string smoothedfileName = m_BaseFolderPath + "/" + "smoothed.ply";
   // pcl::io::savePLYFile(smoothedfileName, *normal_smoothed_targetCloud);

    // meshing of target cloud
    pcl::PolygonMesh mesh;
    ConvertPCDToMesh(normal_smoothed_targetCloud, mesh);
    size_t pos2 = CloudTarget.rfind('.');
    size_t pos1 = CloudTarget.rfind('/');
    string subfile = CloudTarget.substr(pos1 + 1, string::npos);
    string meshname = subfile.substr(0, subfile.rfind('.'));
    std::string meshfileName = m_BaseFolderPath + "/" + meshname + "_mesh.ply";
    LRF_Protocols.ComputePointNormal(mesh);
   // pcl::io::savePLYFile(meshfileName, mesh);
    *normal_smoothed_targetCloud = mesh.cloud;
    /// meshing ends here//////////

    cKDTreeSearch search(true, 12.0f, false);
    cCorrespondenceKdtree cKdTree(search);

    cTransformEstimationSparseICP regis_3d_kdtree;

    // Registration Call
    Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
   // initTransformation = initialPoses[i]; ;
    float norm = 0.5f;
    int numIcpIteration = 100;
    int searchStrategy = 0;
    Eigen::Matrix4f finalMatrix;
    finalMatrix.setIdentity();
    Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
    bool fallback = false;

    regis_3d_kdtree.PrepareSample(CloudSource, normal_smoothed_targetCloud);
    regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
        stop_criteria, fallback, m_diagonalLength);
    CpuTimeProfiler cpuTimeC;
    std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&cKdTree, finalMatrix);
    regis_time = cpuTimeC.GetElapsedSecs();

    std::vector<Eigen::Matrix4f> output_matrix(1);
    finaltransformation = finalMatrix;
    double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
    ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

    //collect data for each view_pair
    string iteration_rmse_time = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + "Iteration_Rmse_new_" + std::to_string(index) + "_" + m_MethodName + ".txt"; //m_DataSetPath
    std::vector<double>rmse = regis_3d_kdtree.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
    string combi_fileName = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + "Time_with_Rmse_new_" + std::to_string(index) + "_" + m_MethodName + ".txt"; //m_DataSetPath
    regis_3d_kdtree.WriteRMSEAndTime(combi_fileName, rmse, error_values);
    error_log("/////////////////Computation End///////////////\n");
    EndDataLog();
    /*i++;
    if (i == 12)
        i = 0;*/
    double corrs_time, optimtime;
    regis_3d_kdtree.GetIcpTimings(corrs_time, optimtime);
    m_corresp_time[index] = corrs_time;
    optim_time[index] = optimtime;
    return regis_time;
}
void NearestNeigborSicp::Evaluate()
{
    int size = this->m_ViewPairs.size();
    string sourceInput = "";
    string targetInput = "";
    string smoothTargetFile = "";
    CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
    CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
    CloudWithoutType normal_smoothed_targetCloud(new pcl::PCLPointCloud2);
    for (int i = 4611; i < 4612; i++)
    {
        std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
        // ReadViewpair for Evaluation 

        ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
        //// load source ply file

        //std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[i].second;
        //iospace::loadPlyFile(inputSourceCloud, sourceCloud);

        Eigen::Vector3f min_pt, max_pt;
        CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
        pcl::fromPCLPointCloud2(*sourceCloud, *tar_new);
        double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
        std::cout << "diagonalLength:" << diagonalLength << std::endl;
        double stop_criteria = ComputeStopCriteria(stop_criteria_multiplier, diagonalLength, this->Error_StdDev);

        // load target pct file
        std::vector <UVData<float>>pixels;
        std::vector<PointType>point_cloud;
        std::vector<float>intensity;

        //// std::string targetPCTCloud = "C:/WorkSpace/Regis_3D/Regis_3D/bimba_0.pct";
        //pct::LoadPCTFIle(parser::target_pctfile_, pixels, intensity, point_cloud);

        // load projection matrix
        std::string projectionFile = "C:/WorkSpace/Regis_3D/Regis_3D/bimba_ProjMat_fov60.txt";
        // Eigen::Matrix4f proj_mat = tool::LoadProjectionmatrix(projectionFile);


        error_log("/////////Computation Begin///////////////////\n");
        // instantiate normal sampling

        size_t sampling_size = sourceCloud->height * sourceCloud->width;
        CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
        cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
        nml_sampling.PrepareSamplingData(sourceCloud);
        CpuTimeProfiler cpuTime;
        nml_sampling.SamplePointCloud(sampledSourceCloud);
        this->m_samplingTime[i] = cpuTime.GetElapsedSecs();

       
        cKDTreeSearch search(true, 12.0f, false);
        cCorrespondenceKdtree cKdTree(search);

        cTransformEstimationSparseICP regis_3d_kdtree;

        // Registration Call
        Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
        initTransformation = InitialPose_Eth[i] ;
        float norm = 0.5f;
        int numIcpIteration = 150;
        int searchStrategy = 0;
        Eigen::Matrix4f finalMatrix;
        finalMatrix.setIdentity();
        Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
        bool fallback = false;

        regis_3d_kdtree.PrepareSample(sampledSourceCloud, targetCloud);
        regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
            stop_criteria, fallback, diagonalLength);
        CpuTimeProfiler cpuTimeC;
        std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&cKdTree, finalMatrix);
        this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
        double corres_time, optim_time;
        regis_3d_kdtree.GetIcpTimings(corres_time, optim_time);
        this->m_corresp_time[i] = corres_time;
        this->optim_time[i] = optim_time;

        std::vector<Eigen::Matrix4f> output_matrix(1);
        this->m_OutPutTransformations[i] = finalMatrix;
        double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
       
        ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");
      
        ////////////GetmergedCloud/////////////////////
        CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
        TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
        CloudWithoutType mergedCloud;

        std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
        std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
        string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
        string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
       // iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

        error_log("/////////////////Computation End///////////////\n");
        EndDataLog();

        /////////RMSE Computation begins//
        if (false == eth_evaluation)
        {
            /////////RMSE Computation begins//
            // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
            //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
            //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
            double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, this->m_ViewPairs[i].second,
                this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
            std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
            RMSEs[i] = rmse_value;
          
        }
        else
        {
            Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], finalMatrix);
            CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
            CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
            RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
            //collect data for each view_pair
            string iteration_rmse_time = m_DataSetPath + "/" + "Iteration_Rmse_" + std::to_string(i) + "_" + m_MethodName + ".txt";
            std::vector<double>rmse = regis_3d_kdtree.GenerateRMSEListforEachViewPair(iteration_rmse_time, sourceCloud, targetCloud);
            string combi_fileName = m_DataSetPath + "/" + "Time_with_Rmse_" + std::to_string(i) + "_" + m_MethodName + ".txt";
            regis_3d_kdtree.WriteRMSEAndTime(combi_fileName, rmse, error_values);
        }
      
    }
}
 double NearestNeigborSicp::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
    string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
    double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
{
    
     // Compute Prinicipal Frame for Filtering
    // int polynomial_degree = 3.0;
    // int mls_iteration = 20;
    // bool with_filter = false;
    // CWendlandWF<float> wendLandWeightF;
    // cMLSCorrespondence mlsEstimateB(targetCloud, wendLandWeightF, m_searchRadius, polynomial_degree, mls_iteration);
    // CpuTimeProfiler cpuTimeD;
    // mlsEstimateB.ComputeMLSSurface();
    // double targ_mlsTime = cpuTimeD.GetElapsedSecs();
    // std::vector<Eigen::Matrix3f>tgt_principal_frame = mlsEstimateB.GetPrincipalFrame();
    // std::vector<Eigen::Vector3f>tar_eig_value = mlsEstimateB.GetEigenValue();
    // targetCloud = mlsEstimateB.GetInputCloud();

    // cKDTreeSearch search(src_principal_frame, tgt_principal_frame, source_eig_value, tar_eig_value, 12.0f, false);
    // search.CreateIndicesVsPointMap(CloudSource, targetCloud);
    //// search.SetFeatureAngleForSourceAndTarget(featureSource, featureTarget);
     cKDTreeSearch search;
     cCorrespondenceKdtree cKdTree(search);
   
    // mlsTime = 0.0;
     // Registration Call
     Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
    // initTransformation = initMatrix ;
     float norm = 0.5f;
     int numIcpIteration = 100;
     int searchStrategy = 0;
     Eigen::Matrix4f finalMatrix;
     finalMatrix.setIdentity();
     Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
     bool fallback = false;

     /*cKDTreeSearch search(true, 12.0f, false);
     cCorrespondenceKdtree cKdTree(search);*/

     cTransformEstimationSparseICP regis_3d_kdtree;
    // static int i = 0;

     //execute pairwise registration
     regis_3d_kdtree.PrepareSample(CloudSource, targetCloud);
     regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
         stop_criteria, fallback, m_diagonalLength);
     CpuTimeProfiler cpuTimeC;
     auto startItr = std::chrono::high_resolution_clock::now();
     std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&cKdTree,
         finalMatrix);
     auto finishItr = std::chrono::high_resolution_clock::now();
     double executeTime = std::chrono::duration_cast<
         std::chrono::duration<double, std::milli>>(finishItr - startItr).count();
     executeTime = executeTime / double(1000);
     std::cout << "registration time:" << executeTime << "sec" << std::endl;
   double regis_time = cpuTimeC.GetElapsedSecs();

     std::vector<Eigen::Matrix4f> output_matrix(1);
     finaltransformation = finalMatrix;
    // double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
    // ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

    // //collect data for each view_pair
    // string iteration_rmse_time = m_DataSetPath + "/" + this->m_OutputMergedDir + "/"  + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt"; //+ this->m_OutputMergedDir + "/"
    //std::vector<double>rmse = regis_3d_kdtree.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
    //string combi_fileName = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
    //regis_3d_kdtree.WriteRMSEAndTime(combi_fileName, rmse,error_values);  // writes time for each individual pair and each iteration
    // error_log("/////////////////Computation End///////////////\n");
    // EndDataLog();
    ///* i++;
    // if (i == 12)
    //     i = 0;*/
    // double corrs_time, optimtime;
    // regis_3d_kdtree.GetIcpTimings(corrs_time, optimtime);
    // m_corresp_time[index] = corrs_time;
    // optim_time[index] = optimtime;
     return regis_time;
}

 void MLSWithFilter::SmoothTargetUsingMLSProjection(std::vector<std::pair<std::string, std::string>> &views,
     string &inputFileName, int index, CloudWithoutType &inputCloud, cMLSCorrespondence &mlsCorres,
     CloudWithoutType &SmoothedNormalCloud)
 {
     if (inputFileName.compare(views.at(index).first) != 0)
     {
         inputFileName = views.at(index).first;
         CloudWithoutType mls_targetCloud = mlsCorres.ComputeMLSSurface();
         SmoothedNormalCloud = mls_targetCloud;
     }
 }
 double MLSWithFilter::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
     string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
     double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
 {
    
     double regis_time = 0.0;
     //Instantiating weight function
     CWendlandWF<float> wendLandWeightF;

     // Smooth Normals using MLS projection
     cMLSCorrespondence mlsEstimate_target(targetCloud, wendLandWeightF, this->m_searchRadius, 3.0f, 20);

     /*CloudWithoutType normal_smoothed_targetCloud(new pcl::PCLPointCloud2);
     CpuTimeProfiler cpuTimeB;
     SmoothTargetUsingMLSProjection(this->m_SimulatedViewPairs, CloudTarget, index, targetCloud,
         mlsEstimate_target, normal_smoothed_targetCloud);
     mlsTime = cpuTimeB.GetElapsedSecs();
     string smoothedfileName = m_BaseFolderPath + "/" + "smoothed.ply";*/
     // pcl::io::savePLYFile(smoothedfileName, *normal_smoothed_targetCloud);

     //// meshing of target cloud
     //pcl::PolygonMesh mesh;
     //ConvertPCDToMesh(normal_smoothed_targetCloud, mesh);
     //size_t pos2 = CloudTarget.rfind('.');
     //size_t pos1 = CloudTarget.rfind('/');
     //string subfile = CloudTarget.substr(pos1 + 1, string::npos);
     //string meshname = subfile.substr(0, subfile.rfind('.'));
     //std::string meshfileName = m_BaseFolderPath + "/" + meshname + "_mesh.ply";
     ////LRF_Protocols.ComputePointNormal(mesh);
     // pcl::io::savePLYFile(meshfileName, mesh);
     //*normal_smoothed_targetCloud = mesh.cloud;
     /// meshing ends here//////////

     // Compute Prinicipal Frame for Filtering
     int polynomial_degree = 3.0;
     int mls_iteration = 20;
     bool with_norm_filter = false;
     bool with_curv_filter = false;
     bool with_tri_projection = false;

    
    /* cMLSCorrespondence mlsEstimateA(CloudSource, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
     CpuTimeProfiler cpuTimeB;
     mlsEstimateA.ComputeMLSSurface();
     mlsTime = cpuTimeB.GetElapsedSecs();
     std::vector<Eigen::Matrix3f>source_principal_frame = mlsEstimateA.GetPrincipalFrame();
     std::vector<Eigen::Vector3f>source_eigen_value = mlsEstimateA.GetEigenValue();
*/
     // instantiate mls search for projection onto target surface
     cMLSearch mls_search(src_principal_frame, source_eig_value, wendLandWeightF,
         m_RadiusScale, polynomial_degree, mls_iteration, m_normThreshold, with_norm_filter, with_curv_filter, with_tri_projection);
    /* CloudSource = mlsEstimateA.GetInputCloud();*/
     mls_search.SetIndicesVsPointMap(Principal_frame_Index_source_map);
    // mls_search.CreateIndicesVsPointMap(CloudSource);
    /* if (with_tri_projection == true)
     {
         mls_search.setPolygonMesh(mesh);
     }*/
   
     //MLS Correspondence Estimator
     cMLSCorrespondence mlsEstimator(mls_search);

     cTransformEstimationSparseICP regis_mlsf;

     // Registration Call
     Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
    //initTransformation = initMatrix;
     float norm = 0.5f;
     int numIcpIteration = 100;
     int searchStrategy = 1;
     Eigen::Matrix4f finalMatrix;
     finalMatrix.setIdentity();
     Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
     bool fallback = m_Fallback;
    
     //execute pairwise registration
     regis_mlsf.PrepareSample(CloudSource, targetCloud); //targetCloud
     regis_mlsf.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
         stop_criteria, fallback, m_diagonalLength);
     CpuTimeProfiler cpuTimeC;
     std::vector<std::pair<double, double>> error_values = regis_mlsf.estimateRigidRegistration(&mlsEstimator,
         finalMatrix);
     regis_time = cpuTimeC.GetElapsedSecs();

     std::vector<Eigen::Matrix4f> output_matrix(1);
     finaltransformation = finalMatrix;
     //double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
     //ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

     ////collect data for each view_pair
     //string iteration_rmse_time = m_DataSetPath + "/" + this->m_OutputMergedDir  + "/" + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
     //std::vector<double>rmse = regis_mlsf.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
     //string combi_fileName = m_DataSetPath + "/" + this->m_OutputMergedDir + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
     //regis_mlsf.WriteRMSEAndTime(combi_fileName, rmse, error_values);
     //error_log("/////////////////Computation End///////////////\n");
     //EndDataLog();
 
     //double corrs_time, optimtime;
     //regis_mlsf.GetIcpTimings(corrs_time, optimtime);
     //m_corresp_time[index] = corrs_time;
     //optim_time[index] = optimtime;
     //error_log("/////////////////Computation End///////////////\n");
     //EndDataLog();
     return regis_time;
 }
 void MLSWithFilter::Evaluate()
 {
     int size = this->m_ViewPairs.size();
     string sourceInput = "";
     string targetInput = "";
     string smoothTargetFile = "";
     CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
     CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
     CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
     for (int i = 0; i < size; i++)
     {
         std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
         // ReadViewpair for Evaluation 

         ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
         //// load source ply file

         Eigen::Vector3f min_pt, max_pt;
         CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
         pcl::fromPCLPointCloud2(*sourceCloud, *tar_new);
         double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
         /*  double eta = (this->Error_StdDev / diagonalLength) * 1.5;
         double stop_criteria = eta * diagonalLength;*/
         double stop_criteria = ComputeStopCriteria(stop_criteria_multiplier, diagonalLength, this->Error_StdDev);


         error_log("/////////Computation Begin///////////////////\n");
         // instantiate normal sampling

         size_t sampling_size = sourceCloud->height * sourceCloud->width;
         CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
         cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
         nml_sampling.PrepareSamplingData(sourceCloud);
         CpuTimeProfiler cpuTime;
         nml_sampling.SamplePointCloud(sampledSourceCloud);
      
         this->m_samplingTime[i] = cpuTime.GetElapsedSecs();

         //Instantiating weight function
         CWendlandWF<float> wendLandWeightF;

         // Compute Prinicipal Frame for Filtering
         int polynomial_degree = 3.0;
         int mls_iteration = 20;
         bool with_filter = true;

         cMLSCorrespondence mlsEstimateA(sampledSourceCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
         CpuTimeProfiler cpuTimeB;
         mlsEstimateA.ComputeMLSSurface();
         this->m_smoothingTime[i] = cpuTimeB.GetElapsedSecs();
         sampledSourceCloud =  mlsEstimateA.GetInputCloud();  // for some points principal frame is not computed and hence filter those points
         
         std::vector<Eigen::Matrix3f>source_principal_frame = mlsEstimateA.GetPrincipalFrame();
         std::vector<Eigen::Vector3f>source_eigen_value = mlsEstimateA.GetEigenValue();
        
         // instantiate mls search for projection onto target surface
         cMLSearch mls_search(source_principal_frame, source_eigen_value, wendLandWeightF,
             m_RadiusScale, polynomial_degree, mls_iteration, m_normThreshold, with_filter);
         mls_search.CreateIndicesVsPointMap(sampledSourceCloud);
         //MLS Correspondence Estimator
         cMLSCorrespondence mlsEstimator(mls_search);

         cTransformEstimationSparseICP regis_mlsf;
         
         // Registration Call
         Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
         initTransformation = initialPoses[i];// InitialPose_Eth[i];
         float norm = 0.5f;
         int numIcpIteration = 100;
         int searchStrategy = 1;
         Eigen::Matrix4f finalMatrix;
         finalMatrix.setIdentity();
         Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
         bool fallback = m_Fallback;

         regis_mlsf.PrepareSample(sampledSourceCloud, targetCloud);
         regis_mlsf.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
             stop_criteria, fallback, diagonalLength);
         CpuTimeProfiler cpuTimeC;
         std::vector<std::pair<double, double>> error_values = regis_mlsf.estimateRigidRegistration(&mlsEstimator, finalMatrix);
         this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
         double corres_time, optim_time;
         regis_mlsf.GetIcpTimings(corres_time, optim_time);
         this->m_corresp_time[i] = corres_time;
         this->optim_time[i] = optim_time;

        // finalMatrix = finalMatrix * InitialPose_Eth[i];
         std::vector<Eigen::Matrix4f> output_matrix(1);
         this->m_OutPutTransformations[i] = finalMatrix;
         double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
         ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

         ////////////GetmergedCloud/////////////////////
      
         TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
         std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
         std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
         string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
         string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
         //iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

         error_log("/////////////////Computation End///////////////\n");
         EndDataLog();

         if (false == eth_evaluation)
         {
             /////////RMSE Computation begins//
             /* std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
             double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
                  sourceCloud, this->m_meanMeshRes);*/  // only for room_dataset;
             double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, this->m_ViewPairs[i].second,
                 this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
             std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
             RMSEs[i] = rmse_value;
         }
         else
         {
             Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], finalMatrix);
             CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
             CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
             RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
             std::cout << "RMSE:" << RMSEs[i] << std::endl;
         }
     }
 }
 void GICPPairwiseRegistration::Evaluate()
 {
     int size = this->m_ViewPairs.size();
     string sourceInput = "";
     string targetInput = "";
     string smoothTargetFile = "";
     CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
     CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
     CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
     CloudWithoutType CloudTarget(new pcl::PCLPointCloud2);
     /*std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[0].first;
     iospace::loadPlyFile(inputTargetCloud, CloudTarget);
     double radius = metric::EstimatePointAvgDistance(CloudTarget);*/
     for (int i = 0; i < size; i++)
     {
         std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
         // ReadViewpair for Evaluation 

         ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
         //// load source ply file


         error_log("/////////Computation Begin///////////////////\n");
         // instantiate normal sampling

         size_t sampling_size = sourceCloud->height * sourceCloud->width;
         CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
         cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
         nml_sampling.PrepareSamplingData(sourceCloud);
         CpuTimeProfiler cpuTime;
         nml_sampling.SamplePointCloud(sampledSourceCloud);
         this->m_samplingTime[i] = cpuTime.GetElapsedSecs();

         // Registration Call
         Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
         initTransformation = initialPoses[i];
         Eigen::Matrix4f finalMatrix;
         finalMatrix.setIdentity();
         Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();

         //transform source cloud by initial transformation
         // CloudWithoutType transformed_source = tool::TransFormationOfCloud(sampledSourceCloud, initTransformation.matrix());

         GICPPointSet p1, p2;
         dgc_transform_t t_base, t0, t1;

         // set up the transformations
         dgc_transform_identity(t_base);
         dgc_transform_identity(t0);
         dgc_transform_identity(t1);

         //set up source cloud into gicp pipeline
         CloudPtr cloud_source(new pcl::PointCloud<PointType>);
         pcl::fromPCLPointCloud2(*sampledSourceCloud, *cloud_source);
         for (size_t iSource = 0; iSource < cloud_source->points.size(); iSource++)
         {
             GICPPoint pt;
             pt.x = cloud_source->points[iSource].x;
             pt.y = cloud_source->points[iSource].y;
             pt.z = cloud_source->points[iSource].z;
             p1.AppendPoint(pt);
         }

         //set up target cloud into gicp pipeline
         CloudPtr cloud_target(new pcl::PointCloud<PointType>);
         pcl::fromPCLPointCloud2(*targetCloud, *cloud_target);
         for (size_t iTarget = 0; iTarget < cloud_target->points.size(); iTarget++)
         {
             GICPPoint pt;
             pt.x = cloud_target->points[iTarget].x;
             pt.y = cloud_target->points[iTarget].y;
             pt.z = cloud_target->points[iTarget].z;
             p2.AppendPoint(pt);
         }
         std::cout << "Building KDTree and computing surface normals/matrices..." << endl;

         p1.SetGICPEpsilon(0.0001);
         p2.SetGICPEpsilon(0.0001);
         p1.BuildKDTree();
         p1.ComputeMatrices();
         p2.BuildKDTree();
         p2.ComputeMatrices();

         for (int r = 0; r < 4; r++)
         {
             for (int c = 0; c < 4; c++)
             {
                 t_base[r][c] = initialPoses[i].matrix()(r, c);
             }
         }
         dgc_transform_copy(t1, t0);

         p2.SetMaxIteration(m_numMaxIter);         // , 8.0 * m_meanMeshRes for LRF data
         m_maxSearchRadius = 8.0 * m_meanMeshRes;  // m_searchRadius is scale multiplier of avg point distance as radius 
         CpuTimeProfiler cpuTimeC;
         int iterations = p2.AlignScan(&p1, t_base, t1, m_maxSearchRadius);
         this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
         // std::cout << "itr:" << iterations << std::endl;
         

         for (int r = 0; r < 4; r++)
         {
             for (int c = 0; c < 4; c++)
             {
                 finalMatrix(r, c) = t1[r][c];
             }
         }
         this->m_OutPutTransformations[i] = finalMatrix * initialPoses[i].matrix();

         ////////////GetmergedCloud/////////////////////

         TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, this->m_OutPutTransformations[i]);
         std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
         std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
         string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
         string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
         //iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

         error_log("/////////////////Computation End///////////////\n");
         EndDataLog();

         if (false == eth_evaluation)
         {
             /////////RMSE Computation begins//
             // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
             //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
             //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
             double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, this->m_OutPutTransformations[i], this->m_ViewPairs[i].second,
                 this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
             std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
             RMSEs[i] = rmse_value;
         }
         else
         {
             Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], this->m_OutPutTransformations[i]);
             CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
             CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, this->m_OutPutTransformations[i]);
             RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
             std::cout << RMSEs[i] << std::endl;
         }
     }
 }

 double GICPPairwiseRegistration::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
     string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
     double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
 {
     // Registration Call
     Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
     // initTransformation = initialPoses[i]; 
     Eigen::Matrix4f finalMatrix;
     finalMatrix.setIdentity();
     Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();

     //transform source cloud by initial transformation
     CloudWithoutType transformed_source = tool::TransFormationOfCloud(CloudSource, initTransformation.matrix());

     GICPPointSet p1, p2;
     dgc_transform_t t_base, t0, t1;

     // set up the transformations
     dgc_transform_identity(t_base);
     dgc_transform_identity(t0);
     dgc_transform_identity(t1);

     //set up source cloud into gicp pipeline
     CloudPtr cloud_source(new pcl::PointCloud<PointType>);
     pcl::fromPCLPointCloud2(*transformed_source, *cloud_source);
     for (size_t iSource = 0; iSource < cloud_source->points.size(); iSource++)
     {
         GICPPoint pt;
         pt.x = cloud_source->points[iSource].x;
         pt.y = cloud_source->points[iSource].y;
         pt.z = cloud_source->points[iSource].z;
         p1.AppendPoint(pt);
     }

     //set up target cloud into gicp pipeline
     CloudPtr cloud_target(new pcl::PointCloud<PointType>);
     pcl::fromPCLPointCloud2(*targetCloud, *cloud_target);
     for (size_t iTarget = 0; iTarget < cloud_target->points.size(); iTarget++)
     {
         GICPPoint pt;
         pt.x = cloud_target->points[iTarget].x;
         pt.y = cloud_target->points[iTarget].y;
         pt.z = cloud_target->points[iTarget].z;
         p2.AppendPoint(pt);
     }
     std::cout << "Building KDTree and computing surface normals/matrices..." << endl;
     p1.SetEpsilon(m_epsilon);
     p2.SetEpsilon(m_epsilon);
     p1.BuildKDTree();
     p1.ComputeMatrices();
     p2.BuildKDTree();
     p2.ComputeMatrices();

     dgc_transform_copy(t1, t0);
     CpuTimeProfiler cpuTimeC;
     p2.SetMaxIteration(m_numMaxIter);
     int iterations = p2.AlignScan(&p1, t_base, t1, m_maxSearchRadius);
     double regis_time = cpuTimeC.GetElapsedSecs();
     std::vector<Eigen::Matrix4f> output_matrix(1);
     
     for (int r = 0; r < 4; r++)
     {
         for (int c = 0; c < 4; c++)
         {
             finalMatrix(r, c) = t1[r][c];
         }
     }
     finaltransformation = finalMatrix;
     error_log("/////////////////Computation End///////////////\n");
     EndDataLog();
     return regis_time;
 }
 void TwoStepBaseline::Evaluate()
 {
     int size = this->m_ViewPairs.size();
     string sourceInput = "";
     string targetInput = "";
     CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
     CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
     CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
     for (int i = 0; i < size; i++)
     {
         std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
         // ReadViewpair for Evaluation 

         ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
       
         Eigen::Vector3f min_pt, max_pt;
         CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
         pcl::fromPCLPointCloud2(*sourceCloud, *tar_new);
         double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
         double stop_criteria = ComputeStopCriteria(stop_criteria_multiplier, diagonalLength, this->Error_StdDev);


         error_log("/////////Computation Begin///////////////////\n");
         // instantiate normal sampling

         size_t sampling_size = sourceCloud->height * sourceCloud->width;
         CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
         cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
         nml_sampling.PrepareSamplingData(sourceCloud);
         CpuTimeProfiler cpuTime;
         nml_sampling.SamplePointCloud(sampledSourceCloud);
         this->m_samplingTime[i] = cpuTime.GetElapsedSecs();


         cKDTreeSearch search(true, 12.0f, false);
         cCorrespondenceKdtree cKdTree(search);

         cTransformEstimationSparseICP regis_3d_combi;

         // Registration Call
         Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
         initTransformation = InitialPose_Eth[i];
         float norm = 0.5f;
         int numIcpIteration = 100;
         int searchStrategy = 0;
         Eigen::Matrix4f semi_finalMatrix;
         semi_finalMatrix.setIdentity();
         Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
         bool fallback = m_Fallback;

         regis_3d_combi.PrepareSample(sampledSourceCloud, targetCloud);
         regis_3d_combi.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
             stop_criteria, fallback, diagonalLength);
       regis_3d_combi.estimateRigidRegistration(&cKdTree, semi_finalMatrix);

         //Instantiating weight function
         CWendlandWF<float> wendLandWeightF;

         int polynomial_degree = 3.0;
         int mls_iteration = 20;
         bool with_filter = false;

         // instantiate mls search for projection onto target surface
         cMLSearch mls_search(targetCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
         //MLS Correspondence Estimator
         cMLSCorrespondence mlsEstimator(mls_search);

       
         // Registration Call
         initTransformation = semi_finalMatrix; // InitialPose_Eth[i];
         searchStrategy = 1;
         Eigen::Matrix4f finalMatrix;
         finalMatrix.setIdentity();
        
         regis_3d_combi.PrepareSample(sampledSourceCloud, targetCloud);
         regis_3d_combi.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
             stop_criteria, fallback, diagonalLength);
         CpuTimeProfiler cpuTimeC;
         std::vector<std::pair<double, double>> error_values = regis_3d_combi.estimateRigidRegistration(&mlsEstimator, finalMatrix);
         this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();



         std::vector<Eigen::Matrix4f> output_matrix(1);
         this->m_OutPutTransformations[i] = finalMatrix;
         double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
         ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

         TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
         std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
         std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
         string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
         string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
         //iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

         error_log("/////////////////Computation End///////////////\n");
         EndDataLog();

         if (false == eth_evaluation)
         {
             /////////RMSE Computation begins//
             //std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
             //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
             //    sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
            double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, this->m_ViewPairs[i].second,
              this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
             std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
             RMSEs[i] = rmse_value;
         }
         else
         {
             Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], finalMatrix);
             CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
             CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
             RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
         }
      
     }
  }
 double TwoStepBaseline::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
     string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
     double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
 {
     mlsTime = 0.0;
     Eigen::Matrix4f semi_finalMatrix;
     semi_finalMatrix.setIdentity();
     // Registration Call
     Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
     // initTransformation = initialPoses[i]; ;
     float norm = 0.5f;
     int numIcpIteration = 100;
     int searchStrategy = 0;
     Eigen::Matrix4f finalMatrix;
     finalMatrix.setIdentity();
     Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
     bool fallback = false;

     cKDTreeSearch search(true, 12.0f, false);
     cCorrespondenceKdtree cKdTree(search);

     cTransformEstimationSparseICP regis_3d_combi;
    // static int i = 0;

     //execute pairwise registration
     regis_3d_combi.PrepareSample(CloudSource, targetCloud);
     regis_3d_combi.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
         stop_criteria, fallback, m_diagonalLength);
     CpuTimeProfiler cpuTimeC1;
     std::vector<std::pair<double, double>> error_values = regis_3d_combi.estimateRigidRegistration(&cKdTree,
         finalMatrix);
     double regis_time1 = cpuTimeC1.GetElapsedSecs();

     semi_finalMatrix = finalMatrix;
    
     //Instantiating weight function
     CWendlandWF<float> wendLandWeightF;

     int polynomial_degree = 3.0;
     int mls_iteration = 20;
     bool with_filter = false;

     // instantiate mls search for projection onto target surface
     cMLSearch mls_search(targetCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
     //MLS Correspondence Estimator
     cMLSCorrespondence mlsEstimator(mls_search);


     // Registration Call
     initTransformation = semi_finalMatrix; // InitialPose_Eth[i];
     searchStrategy = 1;
 

     regis_3d_combi.PrepareSample(CloudSource, targetCloud);
     regis_3d_combi.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
         stop_criteria, fallback, m_diagonalLength);
     CpuTimeProfiler cpuTimeC2;
     regis_3d_combi.estimateRigidRegistration(&mlsEstimator, finalMatrix);
     double regis_time2 = cpuTimeC2.GetElapsedSecs();

     finaltransformation = finalMatrix;
     double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
     ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

     //collect data for each view_pair
     string iteration_rmse_time = m_DataSetPath + "/" + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
     std::vector<double>rmse = regis_3d_combi.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
     string combi_fileName = m_DataSetPath + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
     regis_3d_combi.WriteRMSEAndTime(combi_fileName, rmse, error_values);
     error_log("/////////////////Computation End///////////////\n");
     EndDataLog();
     /*i++;
     if (i == 12)
         i = 0;*/

     error_log("/////////////////Computation End///////////////\n");
     EndDataLog();
     return (regis_time1 + regis_time2);
 }

  void FGRPairWiseRegistration::Evaluate()
 {
      int size = this->m_ViewPairs.size();
      string sourceInput = "";
      string targetInput = "";
      string smoothTargetFile = "";
      CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
      CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
      CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
      CloudWithoutType CloudTarget(new pcl::PCLPointCloud2);
      std::string inputTargetCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[0].first;
     
      for (int i = 962; i < size; i++)
      {
          std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
          // ReadViewpair for Evaluation 

          ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
          //// load source ply file
          // Registration Call
          Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
         initTransformation = InitialPose_Eth[i];
       
          CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
          cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
          nml_sampling.PrepareSamplingData(sourceCloud);
          //CpuTimeProfiler cpuTime;
          nml_sampling.SamplePointCloud(sampledSourceCloud);
         // this->m_samplingTime[i] = cpuTime.GetElapsedSecs();
          //transform source cloud by initial transformation
         CloudWithoutType transformed_source = tool::TransFormationOfCloud(sampledSourceCloud, initTransformation.matrix());
          CloudWithoutType sampledTargetCloud(new pcl::PCLPointCloud2);
          nml_sampling.PrepareSamplingData(targetCloud);
          nml_sampling.SamplePointCloud(sampledTargetCloud);

          error_log("/////////Computation Begin///////////////////\n");
       
   
          CpuTimeProfiler cpuTimeC;
          FPFHfeatures<double> features(targetCloud, max_Radius);
          features.ComputeFeature();
          features.writeFeatures_bin("target.bin");

          features.generateFromCloud(sourceCloud, max_Radius);
          features.ComputeFeature();
          features.writeFeatures_bin("source.bin");

          CApp app;
          app.ReadFeature("target.bin");
          app.ReadFeature("source.bin");
          app.NormalizePoints();
          app.AdvancedMatching();
         
          app.OptimizePairwise(true, ITERATION_NUMBER);
          this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
          this->m_OutPutTransformations[i] = app.GetTrans() *InitialPose_Eth[i];

          ////////////GetmergedCloud/////////////////////

          TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, this->m_OutPutTransformations[i]);
          std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
          std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
          string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
        
          error_log("/////////////////Computation End///////////////\n");
          EndDataLog();

          if (false == eth_evaluation)
          {
              /////////RMSE Computation begins//
              // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
              //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
              //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
              double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, this->m_OutPutTransformations[i], this->m_ViewPairs[i].second,
                  this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
              std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
              RMSEs[i] = rmse_value;
          }
          else
          {
              Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], this->m_OutPutTransformations[i]);
              CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
              CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, this->m_OutPutTransformations[i]);
              RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
              std::cout << RMSEs[i] << std::endl;
          }
      }
 }

  double FGRPairWiseRegistration::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
      string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
      double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
  {
     
      double regis_time = 0.0;
      CpuTimeProfiler cpuTimeC;
      FPFHfeatures<double> features(targetCloud, max_Radius);
      features.ComputeFeature();
      features.writeFeatures_bin("target.bin");

      features.generateFromCloud(CloudSource, max_Radius);
      features.ComputeFeature();
      features.writeFeatures_bin("source.bin");

      CApp app;
      app.ReadFeature("target.bin");
      app.ReadFeature("source.bin");
      app.NormalizePoints();
      app.AdvancedMatching();

      app.OptimizePairwise(true, ITERATION_NUMBER);
      regis_time = cpuTimeC.GetElapsedSecs();
     finaltransformation = app.GetTrans();

      error_log("/////////////////Computation End///////////////\n");
      EndDataLog();
      
      return regis_time;
  }
  double WeightedDepthProjection::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
      string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
      double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
  {
   
      // Registration Call
      Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
      // initTransformation = initialPoses[i]; ;
      float norm = 0.5f;
      int numIcpIteration = 100;
      int searchStrategy = 1;
      Eigen::Matrix4f finalMatrix;
      finalMatrix.setIdentity();
      Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
      bool fallback = false;

      cInterPolation search(m_projectionMatrix,m_TargetPixels);
      cCorrespondenceDepth wdp(search) ;

      cTransformEstimationSparseICP regis_3d_wdp;
      //static int i = 0;

      //execute pairwise registration
      regis_3d_wdp.PrepareSample(CloudSource, targetCloud);
      regis_3d_wdp.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
          stop_criteria, fallback, m_diagonalLength);
      CpuTimeProfiler cpuTimeC;
      std::vector<std::pair<double, double>> error_values = regis_3d_wdp.estimateRigidRegistration(&wdp,
          finalMatrix);
      double regis_time = cpuTimeC.GetElapsedSecs();

      std::vector<Eigen::Matrix4f> output_matrix(1);
      finaltransformation = finalMatrix;
      double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
      ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

      //collect data for each view_pair
      string iteration_rmse_time = m_DataSetPath + "/" + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      std::vector<double>rmse = regis_3d_wdp.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
      string combi_fileName = m_DataSetPath + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      regis_3d_wdp.WriteRMSEAndTime(combi_fileName, rmse, error_values);
      error_log("/////////////////Computation End///////////////\n");
      EndDataLog();
      /*i++;
      if (i == 12)
          i = 0;*/
      double corrs_time, optimtime;
      regis_3d_wdp.GetIcpTimings(corrs_time, optimtime);
      m_corresp_time[index] = corrs_time;
      optim_time[index] = optimtime;
      return regis_time;
  }
  double WeightedPlaneProjection::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
      string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
      double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
  {

      // Registration Call
      Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
      // initTransformation = initialPoses[i]; ;
      float norm = 0.5f;
      int numIcpIteration = 100;
      int searchStrategy = 0;
      Eigen::Matrix4f finalMatrix;
      finalMatrix.setIdentity();
      Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
      bool fallback = false;

      cPlaneProjectionMethod search(m_projectionMatrix, m_TargetPixels);
      cCorrespondencePlaneProjection wpp(search);

      cTransformEstimationSparseICP regis_3d_wdp;
     // static int i = 0;

      //execute pairwise registration
      regis_3d_wdp.PrepareSample(CloudSource, targetCloud);
      regis_3d_wdp.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
          stop_criteria, fallback, m_diagonalLength);
      CpuTimeProfiler cpuTimeC;
      std::vector<std::pair<double, double>> error_values = regis_3d_wdp.estimateRigidRegistration(&wpp,
          finalMatrix);
      double regis_time = cpuTimeC.GetElapsedSecs();

      std::vector<Eigen::Matrix4f> output_matrix(1);
      finaltransformation = finalMatrix;
      double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
      ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

      //collect data for each view_pair
      string iteration_rmse_time = m_DataSetPath + "/" + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      std::vector<double>rmse = regis_3d_wdp.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
      string combi_fileName = m_DataSetPath + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      regis_3d_wdp.WriteRMSEAndTime(combi_fileName, rmse, error_values);
      error_log("/////////////////Computation End///////////////\n");
      EndDataLog();
    /*  i++;
      if (i == 12)
          i = 0;*/
      double corrs_time, optimtime;
      regis_3d_wdp.GetIcpTimings(corrs_time, optimtime);
      m_corresp_time[index] = corrs_time;
      optim_time[index] = optimtime;
      return regis_time;
  }
  void WeightedDepthProjection::Evaluate()
  {

  }

  void WeightedPlaneProjection::Evaluate()
  {

  }
  void CurvatureBasedRegistration::Evaluate()
  {
      int size = this->m_ViewPairs.size();
      string sourceInput = "";
      string targetInput = "";
      string smoothTargetFile = "";
      CloudWithoutType sourceCloud(new pcl::PCLPointCloud2);
      CloudWithoutType targetCloud(new pcl::PCLPointCloud2);
      CloudWithoutType normal_smoothed_targetCloud(new pcl::PCLPointCloud2);
      for (int i = 0; i < size; i++)
      {
          std::cout << "Evaluation pair:" << i + 1 << "/" << size << std::endl;
          // ReadViewpair for Evaluation 

          ReadInputCloudForEvaluation(sourceInput, targetInput, i, sourceCloud, targetCloud);
          //// load source ply file

          //std::string inputSourceCloud = this->m_DataSetPath + "/" + this->m_ViewPairs[i].second;
          //iospace::loadPlyFile(inputSourceCloud, sourceCloud);

          Eigen::Vector3f min_pt, max_pt;
          CloudWithNormalPtr tar_new(new pcl::PointCloud<PointNormalType>);
          pcl::fromPCLPointCloud2(*sourceCloud, *tar_new);
          double  diagonalLength = tool::ComputeOrientedBoundingBoxOfCloud(tar_new, min_pt, max_pt);
          double stop_criteria = ComputeStopCriteria(2.0, diagonalLength, this->Error_StdDev);

         
 
          error_log("/////////Computation Begin///////////////////\n");
       
          CpuTimeProfiler cpuTimeB;
          std::vector<Eigen::Vector3f> egv_s, egv_t;
          ComputeLocalPrincipalFrame(targetInput, sourceInput, i, sourceCloud, targetCloud, egv_s, egv_t);
          this->m_smoothingTime[i] = cpuTimeB.GetElapsedSecs();

          //////////////////feature angle///////////////////////
          featureSource = metric::ComputeFeatureAngleForACloud(sourceCloud, 11);
          featureTarget = metric::ComputeFeatureAngleForACloud(targetCloud, 11);

          // instantiate normal sampling
          size_t sampling_size = sourceCloud->height * sourceCloud->width;
          CloudWithoutType sampledSourceCloud(new pcl::PCLPointCloud2);
          cNormalSpaceSampling nml_sampling(this->m_SampleSize, 0, 20, 20, 20);
          nml_sampling.PrepareSamplingData(sourceCloud);
          CpuTimeProfiler cpuTime;
          nml_sampling.SamplePointCloud(sampledSourceCloud);
          this->m_samplingTime[i] = cpuTime.GetElapsedSecs();

          cKDTreeSearch search(src_principal_frame, tgt_principal_frame, egv_s, egv_t, 12.0f, false);
          search.CreateIndicesVsPointMap(sourceCloud, targetCloud);
          search.SetFeatureAngleForSourceAndTarget(featureSource, featureTarget);
          cCorrespondenceKdtree cKdTree(search);

          cTransformEstimationSparseICP regis_3d_kdtree;

          // Registration Call
          Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
          //initTransformation = initialPoses[i];// initialPoses[i]; ;  // 1) InitialPose_Eth = eth data
          //                                        //   2) initialPoses = LRF
          float norm = 0.5f;
          int numIcpIteration = 100;
          int searchStrategy = 0;
          Eigen::Matrix4f finalMatrix;
          finalMatrix.setIdentity();
          Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
          bool fallback = false;

          regis_3d_kdtree.PrepareSample(sampledSourceCloud, normal_smoothed_targetCloud);
          regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
              stop_criteria, fallback, diagonalLength);
          CpuTimeProfiler cpuTimeC;
          std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&cKdTree, finalMatrix);
          this->m_registration_time[i] = cpuTimeC.GetElapsedSecs();
          double corres_time, optim_time;
          regis_3d_kdtree.GetIcpTimings(corres_time, optim_time);
          this->m_corresp_time[i] = corres_time;
          this->optim_time[i] = optim_time;

          std::vector<Eigen::Matrix4f> output_matrix(1);
          this->m_OutPutTransformations[i] = finalMatrix;
          double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
          ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

          ////////////GetmergedCloud/////////////////////
          CloudWithoutType TransformedSourceCloud(new pcl::PCLPointCloud2);
          TransformedSourceCloud = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
          CloudWithoutType mergedCloud;

          std::string Source_Gt_Base_Name = this->m_ViewPairs[i].second.substr(0, this->m_ViewPairs[i].second.find("."));
          std::string Target_Gt_Base_name = this->m_ViewPairs[i].first.substr(0, this->m_ViewPairs[i].first.find("."));
          string newDirectory = m_DataSetPath + "/" + this->m_OutputMergedDir;
          string mergedFileName = newDirectory + "/" + Source_Gt_Base_Name + "_" + Target_Gt_Base_name + "_merged.ply";
          // iospace::writePlyFile(mergedFileName, TransformedSourceCloud);

          error_log("/////////////////Computation End///////////////\n");
          EndDataLog();

          /////////RMSE Computation begins//
          // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
          //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
          //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
          if (false == eth_evaluation)
          {
              /////////RMSE Computation begins//
              // std::vector<std::pair<std::string, std::string>> fake_pairs = LRF_Protocols.GetFakeNamePairs();
              //double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, fake_pairs[i].first,
              //     sourceCloud, this->m_meanMeshRes);  // only for room_dataset;
              double rmse_value = LRF_Protocols.CalculateRMSEForEvaluationDataSet(this->View_With_GroundTruth, finalMatrix, this->m_ViewPairs[i].second,
                  this->m_ViewPairs[i].first, sourceCloud, this->m_meanMeshRes);
              std::cout << "RMSE:" << std::setprecision(10) << rmse_value << std::endl;
              RMSEs[i] = rmse_value;
          }
          else
          {
              Rot_Trans_Error[i] = tool::GetTransformationError(GroundTruthPoses[i], finalMatrix);
              CloudWithoutType source_gt_transformed = tool::TransFormationOfCloud(sourceCloud, GroundTruthPoses[i]);
              CloudWithoutType source_ft_transformed = tool::TransFormationOfCloud(sourceCloud, finalMatrix);
              RMSEs[i] = ComputeRMSEForEthDataset(source_gt_transformed, source_ft_transformed);
          }
      }
 }

  double CurvatureBasedRegistration::PairWiseRegistration(CloudWithoutType &CloudSource, CloudWithoutType &targetCloud,
      string &CloudTarget, int index, Eigen::Matrix4f & finaltransformation,
      double &mlsTime, double &stop_criteria, Eigen::Affine3f initMatrix)
  {
     
      double regis_time = 0.0;
      //Instantiating weight function
      CWendlandWF<float> wendLandWeightF;

      // Compute Prinicipal Frame for Filtering
      int polynomial_degree = 3.0;
      int mls_iteration = 20;
      bool with_filter = false;

      // computer local reference frame for source cloud
      cMLSCorrespondence mlsEstimateA(CloudSource, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
      CpuTimeProfiler cpuTimeB;
      mlsEstimateA.ComputeMLSSurface();
      mlsTime = cpuTimeB.GetElapsedSecs();
      std::vector<Eigen::Matrix3f>source_principal_frame = mlsEstimateA.GetPrincipalFrame();
      std::vector<Eigen::Vector3f>source_eigen_value = mlsEstimateA.GetEigenValue();
      CloudSource = mlsEstimateA.GetInputCloud();
     

      // computer local reference frame for target cloud
      cMLSCorrespondence mlsEstimateB(targetCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
      CpuTimeProfiler cpuTimeD;
      mlsEstimateB.ComputeMLSSurface();
      double targ_mlsTime = cpuTimeD.GetElapsedSecs();
      std::vector<Eigen::Matrix3f>target_principal_frame = mlsEstimateB.GetPrincipalFrame();
      std::vector<Eigen::Vector3f>target_eigen_value = mlsEstimateB.GetEigenValue();
      targetCloud = mlsEstimateB.GetInputCloud();
    
      // instantiate mls search for projection onto target surface
      cMLSearch mls_search(source_principal_frame, source_eigen_value, wendLandWeightF,
          m_RadiusScale, polynomial_degree, mls_iteration, 0.7, true);
   
      mls_search.CreateIndicesVsPointMap(CloudSource);
      mls_search.SetFeatureAngleForSourceCloud(featureSource);

      //MLS Correspondence Estimator
      cMLSCorrespondence mlsEstimator(mls_search);

      // Registration Call
      Eigen::Affine3f initTransformation = Eigen::Affine3f::Identity();
      float norm = 0.5f;
      int numIcpIteration = 100;
      int searchStrategy = 1;
      Eigen::Matrix4f finalMatrix;
      finalMatrix.setIdentity();
      Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();
      bool fallback = true;
     
      /*cKDTreeSearch search(source_principal_frame, target_principal_frame, source_eigen_value, target_eigen_value, 12.0f, false);
      search.CreateIndicesVsPointMap(CloudSource, targetCloud);
      search.SetFeatureAngleForSourceAndTarget(featureSource, featureTarget);
      cCorrespondenceKdtree cKdTree(search);*/

      cTransformEstimationSparseICP regis_3d_kdtree;

      //execute pairwise registration
      regis_3d_kdtree.PrepareSample(CloudSource, targetCloud);
      regis_3d_kdtree.setSparseICPParameter(initTransformation, norm, numIcpIteration, searchStrategy, gt,
          stop_criteria, fallback, m_diagonalLength);
      CpuTimeProfiler cpuTimeC;
      std::vector<std::pair<double, double>> error_values = regis_3d_kdtree.estimateRigidRegistration(&mlsEstimator,
          finalMatrix);
      regis_time = cpuTimeC.GetElapsedSecs();

      std::vector<Eigen::Matrix4f> output_matrix(1);
      finaltransformation = finalMatrix;
      double std_dev = ETH_Protocols.ComputeStandardDeviation(error_values);
      ExtractStandardDeviation(std_dev, "standard_deviation_error.txt");

      //collect data for each view_pair
      string iteration_rmse_time = m_DataSetPath + "/" + "Iteration_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      std::vector<double>rmse = regis_3d_kdtree.GenerateRMSEListforEachViewPair(iteration_rmse_time, m_sourceGroundTruth, m_targetGroundTruth);
      string combi_fileName = m_DataSetPath + "/" + "Time_with_Rmse_new" + std::to_string(index) + "_" + m_MethodName + ".txt";
      regis_3d_kdtree.WriteRMSEAndTime(combi_fileName, rmse, error_values);
      error_log("/////////////////Computation End///////////////\n");
      EndDataLog();
     
      double corrs_time, optimtime;
      regis_3d_kdtree.GetIcpTimings(corrs_time, optimtime);
      m_corresp_time[index] = corrs_time;
      optim_time[index] = optimtime;
      error_log("/////////////////Computation End///////////////\n");
      EndDataLog();
      return regis_time;
  }
  void CurvatureBasedRegistration::ComputeLocalPrincipalFrame(string &inputTargetFileName,
      string &inputFileName, int index, CloudWithoutType &inputSourceCloud, CloudWithoutType &inputtargetCloud, std::vector<Eigen::Vector3f>&egv_s,
      std::vector<Eigen::Vector3f>&egv_t)
  {
      //Instantiating weight function
      CWendlandWF<float> wendLandWeightF;

      // Compute Prinicipal Frame for Filtering
      int polynomial_degree = 3.0;
      int mls_iteration = 20;
      if (inputFileName.compare(m_ViewPairs.at(index).second) != 0)
      {
          // computer local reference frame for source cloud
          cMLSCorrespondence mlsEstimateA(inputSourceCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
          mlsEstimateA.ComputeMLSSurface();
          src_principal_frame = mlsEstimateA.GetPrincipalFrame();
          std::vector<Eigen::Vector3f>egv_s = mlsEstimateA.GetEigenValue();
          inputSourceCloud = mlsEstimateA.GetInputCloud();
      }
      if (inputTargetFileName.compare(m_ViewPairs.at(index).first) != 0)
      {
          cMLSCorrespondence mlsEstimateB(inputtargetCloud, wendLandWeightF, m_RadiusScale, polynomial_degree, mls_iteration);
          CpuTimeProfiler cpuTimeD;
          mlsEstimateB.ComputeMLSSurface();
          double targ_mlsTime = cpuTimeD.GetElapsedSecs();
          tgt_principal_frame = mlsEstimateB.GetPrincipalFrame();
          egv_t = mlsEstimateB.GetEigenValue();
          inputtargetCloud = mlsEstimateB.GetInputCloud();
      }
  }