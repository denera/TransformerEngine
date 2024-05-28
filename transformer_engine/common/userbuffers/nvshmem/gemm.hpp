#ifndef __TE_NVSHMEM_GEMM_HPP__
#define __TE_NVSHMEM_GEMM_HPP__

#include <cublasLt.h>
#include <cstdint>

#include "macros.hpp.inc"
#include "cublas_helpers.hpp"

template<typename TA, typename TB, typename TC>
class gemm_t {
    
private:
    cublasLtHandle_t                handle;
    cublasLtMatmulDesc_t            operationDesc;
    cublasLtMatrixLayout_t          Adesc;
    cublasLtMatrixLayout_t          Bdesc;
    cublasLtMatrixLayout_t          Cdesc;
    cublasLtMatmulPreference_t      preference;
    cublasLtMatmulHeuristicResult_t heuristicResult;

    size_t workspace_size_B;
    void* workspace;

    size_t m, n, k;

    int32_t sm_count;
    int32_t row_splits;
    int32_t col_splits;
    
public:
    gemm_t(size_t m, size_t n, size_t k, cublasOperation_t transa, cublasOperation_t transb, int32_t sm_count = 0, int32_t row_splits = 0, int32_t col_splits = 0, int32_t* atomics = nullptr) :
        m(m), n(n), k(k), sm_count(sm_count),
        row_splits(row_splits),
        col_splits(col_splits)
    {

        CUBLAS_CHECK(cublasLtCreate(&handle));

        const size_t lda = transa == CUBLAS_OP_N ? m : k;
        const size_t ldb = transb == CUBLAS_OP_N ? k : n;
        const size_t ldc = m;

        heuristicResult = {};

        // Create matrix descriptors. Not setting any extra attributes.
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cublas_type_map<TA>(),
                                                transa == CUBLAS_OP_N ? m : k,
                                                transa == CUBLAS_OP_N ? k : m,
                                                lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cublas_type_map<TB>(),
                                                transb == CUBLAS_OP_N ? k : n,
                                                transb == CUBLAS_OP_N ? n : k,
                                                ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, cublas_type_map<TC>(), 
                                                m, 
                                                n, 
                                                ldc));

        CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        if( is_fp8<TA>() || is_fp8<TB>() ) {
            int8_t yes = 1;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &yes, sizeof(yes)));
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        // Set math SM count
        if (sm_count != 0) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm_count, sizeof(sm_count)));
        }
        
        // Row/col split and consumer relationship
        if(row_splits > 0) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, 
                                                        CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS,
                                                        &row_splits, sizeof(row_splits)));
        }
        if(col_splits > 0) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, 
                                                        CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS,
                                                        &col_splits, sizeof(col_splits)));
        }
        // Only producer for now
        if(atomics != nullptr) {
            ASSERT(row_splits >= 1 && col_splits >= 1);
            ASSERT(row_splits == 1 || col_splits == 1);
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER,
                                                        &atomics, sizeof(atomics)));
        }

        workspace_size_B = 32 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size_B));

        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size_B, sizeof(workspace_size_B)));


        int returnedResults = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, 
                                                     Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult,
                                                     &returnedResults));
        ASSERT(heuristicResult.workspaceSize <= workspace_size_B);
        ASSERT_EQ(returnedResults, 1);
   
    }
        
    void execute(const TA* A, const TB* B, TC* C, int32_t* atomics, cudaStream_t stream) const {

        ASSERT(workspace_size_B != 0);

        if(atomics != nullptr) {
            ASSERT(row_splits >= 1 && col_splits >= 1);
            ASSERT(row_splits == 1 || col_splits == 1);
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER,
                                                        &atomics, sizeof(atomics)));
        }

        float one = 1.0;
        float zero = 0.0;

        CUBLAS_CHECK(cublasLtMatmul(handle,
                                    operationDesc,
                                    static_cast<const void*>(&one),         /* alpha */
                                    static_cast<const void*>(A),            /* A */
                                    Adesc,
                                    static_cast<const void*>(B),            /* B */
                                    Bdesc,
                                    static_cast<const void*>(&zero),        /* beta */
                                    static_cast<void*>(C),                  /* C */
                                    Cdesc,
                                    static_cast<void*>(C),                  /* D == C */
                                    Cdesc,
                                    &heuristicResult.algo,                  /* algo */
                                    workspace,                              /* workspace */
                                    workspace_size_B,                       
                                    stream));                               /* stream */

    }

    void execute(const TA* A, const TB* B, TC* C, cudaStream_t stream) const {
        execute(A, B, C, nullptr, stream);
    }

    template<typename tA, typename tB, typename tC>
    void execute(const tA& A, const tB& B, tC& C, cudaStream_t stream) const {
        ASSERT(A.size() == m * k);
        ASSERT(B.size() == k * n);
        ASSERT(C.size() == m * n);
        this->execute(A.data(), B.data(), C.data(), stream);
    }

    ~gemm_t() {

        if(handle != nullptr) {
            CUBLAS_CHECK(cublasLtDestroy(handle));
        }
        if(preference != nullptr) {
            CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        }
        if(Bdesc != nullptr) {
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        }
        if(Adesc != nullptr) {
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        }
        if(Cdesc != nullptr) {
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        }
        if(operationDesc != nullptr) {
            CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
        }
        if(workspace != nullptr) {
            CUDA_CHECK(cudaFree(workspace));
        }
    }

    gemm_t(const gemm_t<TA, TB, TC>&) = delete;
    gemm_t<TA, TB, TC>& operator=(const gemm_t<TA, TB, TC>&) = delete;

    gemm_t(gemm_t<TA, TB, TC>&& that) {
        this->handle = that.handle;
        this->operationDesc = that.operationDesc;
        this->Adesc = that.Adesc;
        this->Bdesc = that.Bdesc;
        this->Cdesc = that.Cdesc;
        this->preference = that.preference;
        this->heuristicResult = that.heuristicResult;
        this->workspace_size_B = that.workspace_size_B;
        this->workspace = that.workspace;
        this->m = that.m;
        this->n = that.n;
        this->k = that.k;
        this->sm_count = that.sm_count;
        this->row_splits = that.row_splits;
        this->col_splits = that.col_splits;
        that.handle = nullptr;
        that.operationDesc = nullptr;
        that.Adesc = nullptr;
        that.Bdesc = nullptr;
        that.Cdesc = nullptr;
        that.preference = nullptr;
        that.heuristicResult = {};
        that.workspace_size_B = 0;
        that.workspace = nullptr;
        that.m = 0;
        that.n = 0;
        that.k = 0;
        that.sm_count = 0;
        that.row_splits = 0;
        that.col_splits = 0;
    }
    
    gemm_t<TA, TB, TC>& operator=(gemm_t<TA, TB, TC>&& that) {
        std::swap(this->handle, that.handle);
        std::swap(this->operationDesc, that.operationDesc);
        std::swap(this->Adesc, that.Adesc);
        std::swap(this->Bdesc, that.Bdesc);
        std::swap(this->Cdesc, that.Cdesc);
        std::swap(this->preference, that.preference);
        std::swap(this->heuristicResult, that.heuristicResult);
        std::swap(this->workspace_size_B, that.workspace_size_B);
        std::swap(this->workspace, that.workspace);
        std::swap(this->m, that.m);
        std::swap(this->n, that.n);
        std::swap(this->k, that.k);
        std::swap(this->sm_count, that.sm_count);
        std::swap(this->row_splits, that.row_splits);
        std::swap(this->col_splits, that.col_splits);
        return *this;
    }

};

#endif // __TE_NVSHMEM_GEMM_HPP__
