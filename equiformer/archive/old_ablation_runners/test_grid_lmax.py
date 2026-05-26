#!/usr/bin/env python3
"""
测试grid_resolution和lmax的关系
"""
import sys
sys.path.append('src')

def test_grid_resolution_lmax():
    """测试不同grid_resolution和lmax组合"""
    print("=== 测试grid_resolution和lmax关系 ===")
    
    try:
        from enhanced_equiformer_v2 import EnhancedEquiformerV2
        
        # 测试不同的组合
        test_configs = [
            {"grid_resolution": 6, "lmax_list": [2]},   # 刚好满足要求
            {"grid_resolution": 8, "lmax_list": [2]},   # 应该满足
            {"grid_resolution": 10, "lmax_list": [2]},  # 应该满足
            {"grid_resolution": 12, "lmax_list": [2]},  # 应该满足
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n测试配置 {i+1}: grid_resolution={config['grid_resolution']}, lmax_list={config['lmax_list']}")
            try:
                model = EnhancedEquiformerV2(
                    num_layers=2,
                    sphere_channels=64,
                    num_heads=4,
                    grid_resolution=config['grid_resolution'],
                    edge_channels=64,
                    lmax_list=config['lmax_list']
                )
                print(f"✓ 配置 {i+1} 成功")
                
                # 计算参数量
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  参数量: {total_params:,}")
                
            except Exception as e:
                print(f"✗ 配置 {i+1} 失败: {e}")
                
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grid_resolution_lmax()
