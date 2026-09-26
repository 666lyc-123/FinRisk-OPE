# GitHub 上传与 camera-ready 完成步骤

## 1. 上传公开包

在 GitHub 新建一个公开仓库，建议仓库名为 `FinRisk-OPE`，然后把本文件夹中的全部内容上传到仓库根目录。不要只上传 ZIP 文件，否则网页端无法直接展示 README 和代码。

## 2. 检查真实仓库地址

当前包已经写入真实仓库地址：

```text
https://github.com/666lyc-123/FinRisk-OPE
```

上传前请确认 `paper_source/main.tex` 中仍然保留这个地址，并且没有 `PUBLIC_REPOSITORY_URL` 占位符。

## 3. 重新编译论文

```powershell
cd paper_source
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

确认 PDF 中已经显示 GitHub 地址，并且全文仍为 8 页以内。当前包中的 `paper_source/main_preupload.pdf` 已按这个源文件编译。

## 4. GitHub 自动复现

上传后进入仓库的 **Actions** 页面，运行 `reproduce-finrisk-ope`。工作流会安装锁定依赖、执行完整实验、重新生成证据和表格，并执行一致性验证。

## 5. Camera-ready 提交

将含真实 GitHub 地址的新 PDF 送交 IEEE PDF eXpress 验证。通过后，在 ICDMW 的 CPS author kit 中上传 PDF eXpress 认可的文件，并完成版权和作者注册。

当前 `paper_source/main_preupload.pdf` 是上传前候选 PDF，仍需先通过 IEEE PDF eXpress，再提交 CPS。
