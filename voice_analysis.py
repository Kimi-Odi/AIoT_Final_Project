import librosa
import numpy as np

def analyze_voice(audio_file):
    """
    分析給定的音訊檔案，提取音調、音量和語速特徵。

    :param audio_file: 音訊檔案的路徑或類檔案物件。
    :return: 一個包含可讀回饋的字典。
    """
    try:
        # 載入音訊檔案
        y, sr = librosa.load(audio_file, sr=None)

        # --- 1. 音調分析 (Pitch) ---
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_f0 = f0[voiced_flag]
        
        pitch_feedback = "無法偵測到足夠的音高資訊。"
        if voiced_f0.size > 0:
            avg_pitch = np.mean(voiced_f0)
            std_pitch = np.std(voiced_f0)
            
            if std_pitch < 15:
                pitch_feedback = f"您的音調非常平穩 (平均音高: {avg_pitch:.2f} Hz)，可以嘗試增加一些語氣變化，讓表達更生動。"
            elif std_pitch < 30:
                pitch_feedback = f"您的音調相對穩定 (平均音高: {avg_pitch:.2f} Hz)，聽起來很沉穩。"
            else:
                pitch_feedback = f"您的音調變化豐富 (平均音高: {avg_pitch:.2f} Hz)，充滿表現力。"

        # --- 2. 音量分析 (Volume/Energy) ---
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)
        std_rms = np.std(rms)

        volume_feedback = "無法偵測到足夠的音量資訊。"
        if avg_rms > 0:
            # 將標準差正規化為平均值的百分比，使其更具可比性
            normalized_std_rms = std_rms / avg_rms
            
            if normalized_std_rms < 0.2:
                volume_feedback = f"您的音量非常一致 (平均能量: {avg_rms:.3f})，展現了穩定的自信。"
            elif normalized_std_rms < 0.4:
                volume_feedback = f"您的音量控制良好 (平均能量: {avg_rms:.3f})，吐字清晰。"
            else:
                volume_feedback = f"您的音量起伏較大 (平均能量: {avg_rms:.3f})，富有動感，但請注意音量突然變小可能讓面試官聽不清。"

        # --- 3. 語速分析 (Speech Rate) ---
        # 使用能量閾值來區分靜音和語音
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        speech_duration = sum(librosa.get_duration(y=y[start:end], sr=sr) for start, end in non_silent_intervals)
        total_duration = librosa.get_duration(y=y, sr=sr)

        speech_rate_feedback = "無法偵測到足夠的語音活動。"
        if total_duration > 0:
            speech_ratio = speech_duration / total_duration
            
            if speech_ratio < 0.4:
                speech_rate_feedback = f"您的談話中停頓較多 (語音佔比: {speech_ratio:.1%})，可以嘗試讓表達更流暢一些。"
            elif speech_ratio < 0.7:
                speech_rate_feedback = f"您的語速和停頓掌握得很好 (語音佔比: {speech_ratio:.1%})，節奏適中。"
            else:
                speech_rate_feedback = f"您的表達非常緊湊，語速較快 (語音佔比: {speech_ratio:.1%})，請確保面試官能跟上您的思路。"

        return {
            "pitch": pitch_feedback,
            "volume": volume_feedback,
            "speech_rate": speech_rate_feedback,
        }

    except Exception as e:
        return {
            "error": f"處理音訊時發生錯誤: {e}"
        }

