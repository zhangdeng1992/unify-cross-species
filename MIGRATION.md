# 项目拆分记录

来源：https://github.com/zhangdeng1992/codex ，提交 `f191bd4`。

- Unify 三个入口脚本及相关测试、依赖、说明独立保存。
- 输入检查抽取为 `unify_inputs.py`，不再导入 `saturn_integrate.py`。
- 原 `UNIFY.md` 改为本项目 `README.md`，命令路径改为 `unify-cross-species/`。
- 官方 Unify 训练仓库仍作为独立外部依赖，固定版本见 README。
- 发布完成前保留源仓库文件，以便继续使用原有流程。
