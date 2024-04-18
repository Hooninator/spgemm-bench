

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <algorithm>


#include <mpi.h>


#include "CombBLAS/CombBLAS.h"


using namespace combblas;

#define THREADED

template <typename IT, typename NT, typename DER>
class FeatureExtractor{

public:

    typedef PlusTimesSRing<NT, NT> PTTF;

    FeatureExtractor(std::string& matpathA, std::string& matpathB, bool permute)
    {
        std::string matnameA = ExtractMatName(matpathA);
	    std::string matnameB = ExtractMatName(matpathB);

        std::string fname(matnameA+"x"+matnameB);

        if (permute)
            fname += "-permute.out";
        else
            fname += ".out";
        
        outfile.open(fname, std::ofstream::out);
    }


	std::string ExtractMatName(const std::string& path) 
    {
		size_t start = path.rfind('/') + 1; // +1 to start after '/'
		size_t end = path.rfind('.');
		std::string fileName = path.substr(start, end - start);
		return fileName;
	}
    

    void ExtractFeatures(DER& A, DER& B)
    {
        
        std::string Astr("A");
        std::string Bstr("B");

        /* NNZ, M, N */
        GetGlobalStats(A, Astr);
        GetGlobalStats(B, Bstr);


        /* Average, min, max, variance nnz per col */
        GetColStats(A, Astr);
        GetColStats(B, Bstr);


        /* FLOPS, NNZ of output*/
        IT lenStatArr = B.getncol();

        IT * flopCArr = estimateFLOP((A), (B));
        std::string flopName("FlopC");
        GetOutputStats(flopCArr, lenStatArr, flopName);
        
        IT * nnzCArr = estimateNNZ_Hash(A, B, flopCArr);
        std::string nnzName("NnzC");
        GetOutputStats(nnzCArr, lenStatArr, nnzName);


        /* Multiply */
        GetMultTime(A, B);

    }


    void GetGlobalStats(DER& mat, std::string& suffix)
    {
        features.emplace("NNZ-"+suffix, std::to_string(mat.getnnz()));
        features.emplace("M-"+suffix, std::to_string(mat.getnrow()));
        features.emplace("N-"+suffix, std::to_string(mat.getncol()));
    }


    void GetColStats(DER& mat, std::string& suffix)
    {
        auto minColNnz = ReduceMin(mat.begcol(), mat.endcol()); 
        auto maxColNnz = ReduceMax(mat.begcol(), mat.endcol()); 
        auto meanColNnz = ReduceMean(mat.begcol(), mat.endcol()); 
        auto varColNnz = ReduceVariance(mat.begcol(), mat.endcol());

        features.emplace("MinColNnz-"+suffix, std::to_string(minColNnz));
        features.emplace("MaxColNnz-"+suffix, std::to_string(maxColNnz));
        features.emplace("MeanColNnz-"+suffix, std::to_string(meanColNnz));
        features.emplace("VarianceColNnz-"+suffix, std::to_string(varColNnz));
    }


    void GetOutputStats(IT * statArr, IT lenStatArr, std::string& name)
    {
        std::vector<IT> statArrVec(statArr, statArr+lenStatArr);

        IT min = std::reduce(statArrVec.begin(), statArrVec.end(), std::numeric_limits<IT>::min(),
                                    [](IT a, IT b){return std::min(a,b);}); 
        IT max = std::reduce(statArrVec.begin(), statArrVec.end(), IT(0), 
                                    [](IT a, IT b){return std::max(a,b);}); 

        IT total = std::reduce(statArrVec.begin(), statArrVec.end(), IT(0));
        double mean = (double)total / (double)lenStatArr;
        double var = std::reduce(statArrVec.begin(), statArrVec.end(), 0.0,
                            [mean](double sum, IT curr)
                            {
                                sum += std::pow((double)curr - mean, 2);
                                return sum;
                            }) / (lenStatArr - 1);


        features.emplace("Min"+name, std::to_string(min));
        features.emplace("Max"+name, std::to_string(max));
        features.emplace("Total"+name, std::to_string(total));
        features.emplace("Mean"+name, std::to_string(mean));
        features.emplace("Var"+name, std::to_string(var));

    }


    void GetMultTime(DER& A, DER& B)
    {
        double stime = MPI_Wtime();
        LocalHybridSpGEMM<PTTF, NT>(A, B, false, false);
        double etime = MPI_Wtime();
        double totalTime = (etime - stime);
        features.emplace("MultTime", std::to_string(totalTime));
    }


    void WriteSample()
    {

        for (auto const& pair : features)
        {
            outfile<<pair.first+":"+pair.second+"  ";
        }

    }

    
    template <typename T>
    IT ReduceMin(T begin, T end)
    {
        IT curr = std::numeric_limits<IT>::max();
        for (auto colIter = begin; colIter != end; colIter++)
        {
            curr = std::min(curr, colIter.nnz());
        }
    }

    template <typename T>
    IT ReduceMax(T begin, T end)
    {
        IT curr = 0; 
        for (auto colIter = begin; colIter != end; colIter++)
        {
            curr = std::max(curr, colIter.nnz());
        }
    }

    template <typename T>
    double ReduceMean(T begin, T end)
    {
        IT sum = 0;
        IT len = 0;
        for (auto colIter = begin; colIter != end; colIter++)
        {
            sum += colIter.nnz();
            len++;
        }
        return static_cast<double>(sum) / static_cast<double>(len);
    }

    template <typename T>
    double ReduceVariance(T begin, T end)
    {
        auto mean = ReduceMean(begin, end);

        double diff = 0;

        IT len = 0;

        for (auto colIter = begin; colIter != end; colIter++)
        {
            diff += std::pow((double)colIter.nnz() - mean, 2);
            len++;
        }

        return diff / static_cast<double>(len - 1);
    }

    ~FeatureExtractor(){outfile.close();}

    
private:

    std::map<std::string, std::string> features;

    std::ofstream outfile;

};



