


#include "FeatureExtractor.hpp"

using namespace combblas;

int main(int argc, char ** argv)
{

    /* ./spgemm-bench </path/to/matA> </path/to/matB> <permute>*/
	
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    std::string matpathA(argv[1]);
    std::string matpathB(argv[2]);

    bool permute = (bool)(std::atoi(argv[3]));


    std::shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));

    typedef int64_t IT;
    typedef double NT;
    typedef SpDCCols<IT,NT> DER;

    SpParMat<IT,NT,DER> A(grid);
    SpParMat<IT,NT,DER> B(grid);
    A.ParallelReadMM(matpathA, true, maximum<double>());
    B.ParallelReadMM(matpathB, true, maximum<double>());
    if (permute) {
        FullyDistVec<IT,NT> p(A.getcommgrid());
        p.iota(A.getnrow(), 0);
        p.RandPerm();
        (B)(p,p,true);
        matpathB += std::string("-permute");
    }

    {
        FeatureExtractor<IT, NT, DER> extractor(matpathA, matpathB, permute);

        extractor.ExtractFeatures(*(A.seqptr()), *(B.seqptr()));
        
        extractor.WriteSample();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //MPI_Finalize();
    //Don't ask

    return 0;
}



