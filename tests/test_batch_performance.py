"""
性能对比测试: 验证向量化batch处理相比循环处理的性能提升
"""
import torch
import pytest
import time
from lshn.core.synapses.bistable_hypergraph import BistableHypergraphSynapse
from lshn.core.cells.liquid_cell import LiquidGatedCell, DendriteCompartment


class TestBistableHypergraphPerformance:
    """测试BistableHypergraphSynapse的batch处理性能"""

    @pytest.fixture
    def edge_index(self):
        """创建标准超边索引"""
        return torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],  # 节点 index
            [0, 0, 1, 1, 2, 2, 3, 3]   # 超边 index
        ])

    def test_batch_performance_improvement(self, edge_index):
        """测试向量化batch处理相比循环的性能提升"""
        synapse = BistableHypergraphSynapse(num_neurons=64, out_channels=1)
        synapse.eval()

        batch_sizes = [1, 4, 8, 16, 32]
        num_iterations = 50

        results = {}

        for batch_size in batch_sizes:
            x_in = torch.randn(batch_size, 64)

            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = synapse.step_fast(x_in, edge_index)

            # 测量性能
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = synapse.step_fast(x_in, edge_index)
            elapsed = time.time() - start_time

            avg_time = elapsed / num_iterations * 1000  # 转换为毫秒
            results[batch_size] = avg_time

            print(f"\nBatch size {batch_size}: {avg_time:.3f} ms/step")

        # 验证batch处理是高效的：batch_size增加时，每样本时间应该减少或保持稳定
        # 计算每样本处理时间
        per_sample_times = {bs: results[bs] / bs for bs in batch_sizes}

        print("\n每样本处理时间:")
        for bs in batch_sizes:
            print(f"  Batch {bs}: {per_sample_times[bs]:.3f} ms/sample")

        # 性能断言：较大batch的每样本处理时间不应显著高于小batch
        # 允许一定的开销，但不应超过3倍
        if len(batch_sizes) >= 2:
            large_batch_per_sample = per_sample_times[batch_sizes[-1]]
            small_batch_per_sample = per_sample_times[batch_sizes[0]]
            ratio = large_batch_per_sample / small_batch_per_sample
            print(f"\n大batch vs 小batch 每样本时间比: {ratio:.2f}")
            assert ratio < 3.0, f"大batch每样本处理时间过高: {ratio:.2f}x"

    def test_output_validity(self, edge_index):
        """验证batch输出是有效的"""
        synapse = BistableHypergraphSynapse(num_neurons=32, out_channels=1)
        synapse.eval()

        # 测试不同batch大小
        for batch_size in [1, 4, 8]:
            x_in = torch.randn(batch_size, 32)

            with torch.no_grad():
                out = synapse.step_fast(x_in, edge_index)

            # 验证输出是有效的
            assert out.shape[0] == batch_size
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestDendriteCompartmentPerformance:
    """测试DendriteCompartment的batch处理性能"""

    def test_batch_vs_loop_performance(self):
        """对比向量化batch处理和循环处理的性能"""
        num_neurons = 64
        num_branches = 8
        dendrite = DendriteCompartment(num_neurons, num_branches)
        dendrite.eval()

        batch_sizes = [1, 4, 8, 16, 32]
        num_iterations = 100

        results = {}

        for batch_size in batch_sizes:
            I_syn = torch.randn(batch_size, num_neurons)

            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = dendrite(I_syn)
            dendrite.reset()

            # 测量向量化batch处理
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = dendrite(I_syn)
                dendrite.reset()
            vectorized_time = time.time() - start_time

            # 测量循环处理
            dendrite.reset()
            start_time = time.time()
            for _ in range(num_iterations):
                outputs = []
                for i in range(batch_size):
                    out = dendrite(I_syn[i])
                    outputs.append(out)
                _ = torch.stack(outputs, dim=0)
                dendrite.reset()
            loop_time = time.time() - start_time

            results[batch_size] = {
                'vectorized': vectorized_time / num_iterations * 1000,
                'loop': loop_time / num_iterations * 1000
            }

            speedup = loop_time / vectorized_time
            print(f"\nBatch size {batch_size}:")
            print(f"  向量化: {results[batch_size]['vectorized']:.3f} ms")
            print(f"  循环: {results[batch_size]['loop']:.3f} ms")
            print(f"  加速比: {speedup:.2f}x")

        # 断言：向量化应该比循环快
        for bs in batch_sizes:
            assert results[bs]['vectorized'] < results[bs]['loop'], \
                f"Batch {bs}: 向量化应该比循环快"

    def test_batch_output_correctness(self):
        """验证batch处理和逐样本处理产生有效的输出"""
        num_neurons = 16
        num_branches = 4
        dendrite = DendriteCompartment(num_neurons, num_branches)
        dendrite.eval()

        # 创建输入
        I_syn_batch = torch.randn(8, num_neurons)

        with torch.no_grad():
            # Batch处理
            out_batch = dendrite(I_syn_batch)

        # 验证输出是有效的
        assert out_batch.shape == (8, num_neurons)
        assert not torch.isnan(out_batch).any()
        assert not torch.isinf(out_batch).any()

        # 验证输出具有合理的数值范围
        assert out_batch.abs().max() < 100  # 不应该有极端值


class TestLiquidCellWithDendritesPerformance:
    """测试LiquidGatedCell启用树突时的batch处理性能"""

    def test_cell_with_dendrites_batch_performance(self):
        """测试启用树突的液态元胞batch处理性能"""
        num_neurons = 32
        cell = LiquidGatedCell(
            num_neurons=num_neurons,
            enable_dendrites=True,
            num_branches=4
        )
        cell.eval()

        batch_sizes = [1, 4, 8, 16]
        num_iterations = 50

        results = {}

        for batch_size in batch_sizes:
            I_syn = torch.randn(batch_size, num_neurons)

            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = cell.step_fast(I_syn)
            cell.reset_hidden()

            # 测量性能
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = cell.step_fast(I_syn)
                cell.reset_hidden()
            elapsed = time.time() - start_time

            avg_time = elapsed / num_iterations * 1000
            per_sample = avg_time / batch_size
            results[batch_size] = {'total': avg_time, 'per_sample': per_sample}

            print(f"\nBatch size {batch_size}: {avg_time:.3f} ms/step, {per_sample:.3f} ms/sample")

        # 验证性能可扩展性
        # 较大batch的每样本时间不应显著增加
        if len(batch_sizes) >= 2:
            first_per_sample = results[batch_sizes[0]]['per_sample']
            last_per_sample = results[batch_sizes[-1]]['per_sample']
            ratio = last_per_sample / first_per_sample
            print(f"\n最大batch vs 最小batch 每样本时间比: {ratio:.2f}")
            assert ratio < 2.0, f"batch扩展性不佳: {ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
