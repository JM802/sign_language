/**
 * 帮助面板 - 对应 PyQt 中的 help_browser
 */
export default function HelpPanel() {
  return (
    <div className="help-panel">
      <h2>👋 欢迎使用手语智瞳</h2>

      <div className="help-section">
        <h3>📷 摄像头模式</h3>
        <ol>
          <li>
            点击左侧 <strong>Camera</strong> 进入实时模式
          </li>
          <li>允许浏览器使用摄像头</li>
          <li>
            点击左下角 <strong>Start</strong> 按钮开始录制手语动作
          </li>
          <li>
            做完动作后点击 <strong>Stop</strong> 按钮停止录制
          </li>
          <li>等待识别结果显示在右侧面板</li>
        </ol>
      </div>

      <div className="help-section">
        <h3>🎬 视频模式</h3>
        <ol>
          <li>
            点击左侧 <strong>Video</strong> 进入视频模式
          </li>
          <li>选择或拖入一个手语视频文件</li>
          <li>系统将自动分析并显示识别结果</li>
        </ol>
      </div>

      <div className="help-section">
        <h3>💡 使用技巧</h3>
        <ul>
          <li>确保光线充足，手部在画面中清晰可见</li>
          <li>动作完整即可停止录制，无需等待太长时间</li>
          <li>
            点击右侧的识别结果气泡可以 <strong>语音朗读</strong>
          </li>
          <li>点击气泡右上角的 × 可以删除该条结果</li>
        </ul>
      </div>

      <div className="help-section">
        <h3>⚠️ 注意事项</h3>
        <ul>
          <li>请使用现代浏览器（Chrome、Edge 等），以确保摄像头正常工作</li>
          <li>首次使用需允许浏览器访问摄像头权限</li>
          <li>本模型需要时序动作数据，暂不支持单张图片识别</li>
        </ul>
      </div>
    </div>
  );
}
