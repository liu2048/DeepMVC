# src/data Directory

This directory contains scripts and modules for data loading, preprocessing, and augmentation.

## Files

- `__init__.py`: Initializes the data module.
- `augmenter.py`: Contains data augmentation functions and classes.
- `data_module.py`: Defines the data module for loading and processing datasets.
- `load.py`: Contains functions for loading datasets.
- `make_dataset.py`: Contains scripts for generating datasets.
- `pair_dataset.py`: Contains scripts for generating paired datasets for pre-training.
- `torchvision_datasets.py`: Contains functions for loading datasets from the torchvision library.

## Toggle Chinese Translation

<button onclick="toggleTranslation()">Toggle Chinese Translation</button>

<script>
function toggleTranslation() {
  var elements = document.getElementsByClassName('chinese-translation');
  for (var i = 0; i < elements.length; i++) {
    if (elements[i].style.display === 'none') {
      elements[i].style.display = 'block';
    } else {
      elements[i].style.display = 'none';
    }
  }
}
</script>

<div class="chinese-translation" style="display:none;">
  <h1>src/data 目录</h1>
  <p>此目录包含用于数据加载、预处理和增强的脚本和模块。</p>
  <h2>文件</h2>
  <ul>
    <li><code>__init__.py</code>：初始化数据模块。</li>
    <li><code>augmenter.py</code>：包含数据增强函数和类。</li>
    <li><code>data_module.py</code>：定义用于加载和处理数据集的数据模块。</li>
    <li><code>load.py</code>：包含加载数据集的函数。</li>
    <li><code>make_dataset.py</code>：包含生成数据集的脚本。</li>
    <li><code>pair_dataset.py</code>：包含生成用于预训练的配对数据集的脚本。</li>
    <li><code>torchvision_datasets.py</code>：包含从 torchvision 库加载数据集的函数。</li>
  </ul>
</div>
