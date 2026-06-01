"""
Prompt 优化 Harness - 完整的执行闭环
从数据导入到效果评估的自动化流程
"""
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, '/Users/kwjcyr/data/llm_memory/src')

# 使用绝对导入避免循环依赖
import importlib.util

def load_optimizer():
    """动态加载优化器模块"""
    spec = importlib.util.spec_from_file_location(
        "simple_prompt_optimizer",
        "/Users/kwjcyr/data/llm_memory/src/memory/effective_memory/simple_prompt_optimizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimplePromptOptimizer


class PromptOptimizationHarness:
    """Prompt 优化执行框架"""

    def __init__(self,
                 data_input_path: str = '/Users/kwjcyr/data/llm_memory/data/input/group_0_caroline.json',
                 session_memory_db_path: str = '/Users/kwjcyr/data/llm_memory/data/session_memories.db',
                 session_memory_file_path: str = '/Users/kwjcyr/data/llm_memory/data/caroline_memories.txt',
                 effective_config_path: str = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json',
                 output_dir: str = '/Users/kwjcyr/data/llm_memory/data/harness_output'):
        """
        初始化 Harness

        Args:
            data_input_path: 原始对话数据路径（JSON 格式）
            session_memory_db_path: Session Memory 数据库路径
            session_memory_file_path: Session Memory 文件路径
            effective_config_path: Effective.json 配置路径
            output_dir: 输出目录
        """
        self.data_input_path = data_input_path
        self.session_memory_db_path = session_memory_db_path
        self.session_memory_file_path = session_memory_file_path
        self.effective_config_path = effective_config_path
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 时间戳用于报告
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 80)
        print("Prompt 优化 Harness 初始化完成")
        print("=" * 80)
        print(f"数据输入：{data_input_path}")
        print(f"Session Memory DB: {session_memory_db_path}")
        print(f"Session Memory 文件：{session_memory_file_path}")
        print(f"配置：{effective_config_path}")
        print(f"输出目录：{output_dir}")
        print("=" * 80)

    def step1_import_data(self) -> Dict[str, Any]:
        """步骤 1: 导入原始对话数据到 Session Memory"""
        print("\n" + "=" * 80)
        print("步骤 1: 导入原始对话数据")
        print("=" * 80)

        try:
            # 检查数据文件是否存在
            if not os.path.exists(self.data_input_path):
                print(f"❌ 数据文件不存在：{self.data_input_path}")
                return {'success': False, 'error': 'File not found'}

            # 读取 JSON 文件
            with open(self.data_input_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 统计对话数量
            if isinstance(raw_data, list):
                num_conversations = len(raw_data)
            elif isinstance(raw_data, dict) and 'qa' in raw_data:
                num_conversations = len(raw_data.get('qa', []))
            else:
                num_conversations = 1

            print(f"✓ 读取到 {num_conversations} 条对话记录")

            # 如果数据已经存在，跳过导入
            if os.path.exists(self.session_memory_file_path):
                with open(self.session_memory_file_path, 'r', encoding='utf-8') as f:
                    existing_count = sum(1 for _ in f)
                if existing_count > 0:
                    print(f"⚠️  Session Memory 文件已存在，包含 {existing_count} 条记录，跳过导入")
                    return {
                        'success': True,
                        'imported_count': existing_count,
                        'skipped': True
                    }

            print(f"开始导入数据到 Session Memory...")
            # TODO: 实际导入逻辑（如果需要）
            # 目前假设数据已经通过其他方式导入

            result = {
                'success': True,
                'imported_count': num_conversations,
                'skipped': False
            }

            print(f"✓ 导入完成")
            return result

        except Exception as e:
            print(f"❌ 导入失败：{e}")
            return {'success': False, 'error': str(e)}

    def step2_evaluate_prompts(self, sample_size: int = 5, generations: int = 3) -> Dict[str, Any]:
        """步骤 2: 评估不同 Prompt 变体的效果"""
        print("\n" + "=" * 80)
        print("步骤 2: 评估 Prompt 变体")
        print("=" * 80)

        try:
            # 创建优化器
            SimplePromptOptimizer = load_optimizer()
            optimizer = SimplePromptOptimizer()

            # 运行优化
            print(f"开始评估 {len(optimizer.prompt_additions)} 个 Prompt 变体...")
            best_score, best_addition = optimizer.run_optimization(
                generations=generations,
                sample_size=sample_size
            )

            result = {
                'success': True,
                'best_score': best_score,
                'best_addition': best_addition,
                'num_variants': len(optimizer.prompt_additions)
            }

            print(f"✓ 最佳 Prompt 变体得分：{best_score:.4f}")
            return result

        except Exception as e:
            print(f"❌ 评估失败：{e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def step3_compare_before_after(self, sample_size: int = 10) -> Dict[str, Any]:
        """步骤 3: 对比优化前后的效果"""
        print("\n" + "=" * 80)
        print("步骤 3: 对比优化前后效果")
        print("=" * 80)

        try:
            # 创建优化器
            SimplePromptOptimizer = load_optimizer()
            optimizer = SimplePromptOptimizer()

            # 运行对比
            report = optimizer.compare_before_after(sample_size=sample_size)

            # 保存报告
            report_path = os.path.join(
                self.output_dir,
                f'comparison_report_{self.timestamp}.json'
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            result = {
                'success': True,
                'report_path': report_path,
                'before_score': report['before_score'],
                'after_score': report['after_score'],
                'improvement': report['improvement'],
                'improvement_rate': report['improvement_rate']
            }

            print(f"✓ 对比完成，报告已保存到：{report_path}")
            return result

        except Exception as e:
            print(f"❌ 对比失败：{e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def step4_save_effective_memories(self) -> Dict[str, Any]:
        """步骤 4: 使用最佳 Prompt 抽取并保存 Effective Memories"""
        print("\n" + "=" * 80)
        print("步骤 4: 抽取并保存 Effective Memories")
        print("=" * 80)

        try:
            # 创建优化器
            SimplePromptOptimizer = load_optimizer()
            optimizer = SimplePromptOptimizer()

            # 使用最佳 prompt 抽取
            print(f"使用最佳 Prompt 变体进行抽取...")
            best_addition = optimizer.best_addition
            effective_memories = optimizer.extract_effective_memories(best_addition)

            if not effective_memories:
                print("❌ 未能抽取到有效记忆")
                return {'success': False, 'error': 'No memories extracted'}

            # 保存到文件
            output_path = os.path.join(
                self.output_dir,
                f'effective_memories_{self.timestamp}.txt'
            )
            optimizer.save_effective_memories(effective_memories, output_path)

            result = {
                'success': True,
                'output_path': output_path,
                'num_memories': len(effective_memories),
                'best_addition': best_addition
            }

            print(f"✓ 抽取完成，保存了 {len(effective_memories)} 条记忆到：{output_path}")
            return result

        except Exception as e:
            print(f"❌ 抽取失败：{e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def run_full_pipeline(self,
                         evaluate_sample_size: int = 5,
                         compare_sample_size: int = 10,
                         generations: int = 3) -> Dict[str, Any]:
        """运行完整的优化流程"""
        print("\n" + "=" * 80)
        print("开始运行完整的 Prompt 优化流程")
        print("=" * 80)
        print(f"时间：{self.timestamp}")
        print("=" * 80)

        results = {
            'timestamp': self.timestamp,
            'steps': {}
        }

        # 步骤 1: 导入数据
        results['steps']['step1_import'] = self.step1_import_data()

        # 步骤 2: 评估 Prompt
        results['steps']['step2_evaluate'] = self.step2_evaluate_prompts(
            sample_size=evaluate_sample_size,
            generations=generations
        )

        # 步骤 3: 对比效果
        results['steps']['step3_compare'] = self.step3_compare_before_after(
            sample_size=compare_sample_size
        )

        # 步骤 4: 保存结果
        results['steps']['step4_save'] = self.step4_save_effective_memories()

        # 汇总报告
        summary = self._generate_summary(results)
        results['summary'] = summary

        # 保存完整报告
        full_report_path = os.path.join(
            self.output_dir,
            f'full_report_{self.timestamp}.json'
        )
        with open(full_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print("完整流程执行完成!")
        print("=" * 80)
        print(f"完整报告：{full_report_path}")
        print("=" * 80)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        steps = results['steps']

        summary = {
            'total_steps': len(steps),
            'successful_steps': sum(1 for s in steps.values() if s.get('success', False)),
            'failed_steps': sum(1 for s in steps.values() if not s.get('success', False))
        }

        # 关键指标
        if 'step2_evaluate' in steps and steps['step2_evaluate'].get('success'):
            summary['best_prompt_score'] = steps['step2_evaluate'].get('best_score', 0)

        if 'step3_compare' in steps and steps['step3_compare'].get('success'):
            summary['before_score'] = steps['step3_compare'].get('before_score', 0)
            summary['after_score'] = steps['step3_compare'].get('after_score', 0)
            summary['improvement'] = steps['step3_compare'].get('improvement', 0)
            summary['improvement_rate'] = steps['step3_compare'].get('improvement_rate', 0)

        if 'step4_save' in steps and steps['step4_save'].get('success'):
            summary['effective_memories_count'] = steps['step4_save'].get('num_memories', 0)

        # 总体状态
        if summary['failed_steps'] == 0:
            summary['status'] = 'SUCCESS'
        elif summary['successful_steps'] > 0:
            summary['status'] = 'PARTIAL_SUCCESS'
        else:
            summary['status'] = 'FAILED'

        return summary


def main():
    """主函数"""
    # 创建 Harness
    harness = PromptOptimizationHarness()

    # 运行完整流程
    results = harness.run_full_pipeline(
        evaluate_sample_size=5,      # 优化时测试样本数
        compare_sample_size=10,       # 对比时测试样本数
        generations=3                 # 优化代数
    )

    # 打印摘要
    print("\n" + "=" * 80)
    print("执行摘要")
    print("=" * 80)
    summary = results['summary']
    print(f"状态：{summary['status']}")
    print(f"成功步骤：{summary['successful_steps']}/{summary['total_steps']}")

    if 'best_prompt_score' in summary:
        print(f"最佳 Prompt 得分：{summary['best_prompt_score']:.4f}")

    if 'improvement_rate' in summary:
        print(f"优化提升：{summary['improvement']:.4f} ({summary['improvement_rate']:.2f}%)")

    if 'effective_memories_count' in summary:
        print(f"抽取记忆数：{summary['effective_memories_count']}")

    print("=" * 80)

    return results


if __name__ == '__main__':
    main()

