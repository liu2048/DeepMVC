# Source Directory

This directory contains the main source code for the project. Below is a brief description of each subdirectory and its purpose.

## Subdirectories

- `config`: Contains configuration files and templates for experiments, datasets, models, etc.
- `data`: Contains scripts and modules for data loading, preprocessing, and augmentation.
- `lib`: Contains utility modules and functions for evaluation, logging, loss computation, etc.
- `models`: Contains the implementation of various models used in the project.

For detailed descriptions of the files in each subdirectory, please refer to the respective `README.md` files in the corresponding directories.

## Toggle Chinese Translation

<button onclick="toggleTranslation()">Toggle Chinese Translation</button>

<script>
function toggleTranslation() {
  var x = document.getElementById("chinese-translation");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
</script>

<div id="chinese-translation" style="display:none;">
  <h1>源目录</h1>
  <p>此目录包含项目的主要源代码。以下是每个子目录及其用途的简要说明。</p>

  <h2>子目录</h2>
  <ul>
    <li><code>config</code>：包含实验、数据集、模型等的配置文件和模板。</li>
    <li><code>data</code>：包含数据加载、预处理和增强的脚本和模块。</li>
    <li><code>lib</code>：包含用于评估、日志记录、损失计算等的实用程序模块和函数。</li>
    <li><code>models</code>：包含项目中使用的各种模型的实现。</li>
  </ul>

  <p>有关每个子目录中文件的详细说明，请参阅相应目录中的<code>README.md</code>文件。</p>
</div>
