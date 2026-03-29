#!/bin/bash

# 最大并发任务数
MAX_JOBS=6

# 每个任务分配的线程数
THREADS_PER_JOB=15

# HISAT2 索引路径
INDEX_PATH="/mnt/disk2/srtp2024/TJW/External_Validation/refs/grch38/genome"

# SAF 注释文件路径
SAF_FILE="/mnt/disk4/srtp2024/TCGA/eRNA_standard_500bp.saf"

# 全局日志路径
GLOBAL_LOG="/mnt/disk4/srtp2024/run_new_saf.log"

# 初始化命名管道，用于并发令牌控制
tmp_fifo="/tmp/$$.fifo"
mkfifo "$tmp_fifo"
exec 6<> "$tmp_fifo"
rm "$tmp_fifo"

# 预填充令牌
for ((i=0; i<MAX_JOBS; i++)); do echo >&6; done

run_dataset() {
    DATA_ROOT=$1
    ANALYSIS_DIR="${DATA_ROOT}/analysis"

    echo "==================================================" | tee -a $GLOBAL_LOG
    echo "[数据集启动] $DATA_ROOT" | tee -a $GLOBAL_LOG

    # 确保分析目录存在
    if [ ! -d "$ANALYSIS_DIR" ]; then
        mkdir -p "$ANALYSIS_DIR"
    fi
    # 不执行 rm -rf，避免误删已有数据

    cd "$ANALYSIS_DIR" || exit

    # 扫描上级目录中的 FASTQ 文件
    FILES=$(find .. -maxdepth 1 -name "*_1.fastq.gz")

    if [ -z "$FILES" ]; then
        echo "[错误] 目录下没有找到 fastq 文件" | tee -a $GLOBAL_LOG
        return
    fi

    echo "[开始并行比对] 并发: $MAX_JOBS, 线程/任务: $THREADS_PER_JOB" | tee -a $GLOBAL_LOG

    # 并行处理每个样本
    for r1 in ../*_1.fastq.gz; do
        # 获取令牌，无令牌时阻塞等待
        read -u6

        {
            filename=$(basename "$r1")
            id=${filename%_1.fastq.gz}
            r2="../${id}_2.fastq.gz"
            sample_log="${id}.log"

            if [[ -f "$r2" ]]; then
                echo "[开始] $id $(date +%H:%M)" > "$sample_log"

                # HISAT2 比对并通过管道直接排序输出 BAM
                /mnt/disk2/srtp2024/miniconda3/envs/bio_pipeline/bin/hisat2 \
                    -p $THREADS_PER_JOB --dta -x $INDEX_PATH \
                    -1 "$r1" -2 "$r2" \
                    --summary-file "${id}.summary.txt" 2>> "$sample_log" \
                | samtools sort -@ 4 -o "${id}.bam" - 2>> "$sample_log"

                if [ -s "${id}.bam" ]; then
                    samtools index -@ 4 "${id}.bam"
                    echo "[完成] $id" >> "$sample_log"
                    echo "$id 完成" >> $GLOBAL_LOG
                else
                    echo "[失败] $id BAM 为空" >> "$sample_log"
                    echo "$id 失败" >> $GLOBAL_LOG
                fi
            else
                echo "[跳过] $id 缺少 R2 文件" >> $GLOBAL_LOG
            fi

            # 归还令牌
            echo >&6
        } &
    done

    # 等待本批次所有后台任务完成
    echo "[等待] 本批次剩余任务运行中..." | tee -a $GLOBAL_LOG
    wait

    # 所有 BAM 就绪后统一运行 FeatureCounts
    echo "[开始定量] FeatureCounts..." | tee -a $GLOBAL_LOG

    BAM_LIST=$(ls *.bam 2>/dev/null)
    if [ -n "$BAM_LIST" ]; then
        /mnt/disk2/srtp2024/miniconda3/envs/bio_pipeline/bin/featureCounts \
            -T 48 -p -O --minOverlap 10 \
            -a "$SAF_FILE" -F SAF \
            -o "counts_matrix_500bp.txt" \
            $BAM_LIST >> $GLOBAL_LOG 2>&1

        cut -f 1,7- counts_matrix_500bp.txt > counts_matrix_500bp_clean.txt
        echo "[定量完成] counts_matrix_500bp.txt" | tee -a $GLOBAL_LOG
    else
        echo "[错误] 未找到 BAM 文件，跳过定量" | tee -a $GLOBAL_LOG
    fi
}

# ==================== 主执行流程 ====================
echo "=== 任务开始: $(date) ===" > $GLOBAL_LOG

# 依次处理两个数据集
run_dataset "/mnt/disk4/srtp2024/GSE229571"
run_dataset "/mnt/disk4/srtp2024/GSE225846"

# 关闭令牌管道
exec 6>&-
echo "=== 所有任务完成: $(date) ===" | tee -a $GLOBAL_LOG
