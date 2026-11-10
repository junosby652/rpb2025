"""Generate Korean slides and an experiment report from measured results."""
import argparse
import json
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def generate(root):
    root = Path(root)
    data = json.loads((root / "evaluation.json").read_text())
    configs = {n: json.loads((root / n / "config.json").read_text()) for n in ["baseline", "improved"]}
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    def text(slide, content, x, y, w, h, size=24, color="E2E8F0", bold=False):
        tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        tf.word_wrap = True
        for i, line in enumerate(content.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name, p.font.size = "Noto Sans CJK KR", Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.space_after = Pt(18)
        return tf

    def slide(title, subtitle=""):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = RGBColor.from_string("0F172A")
        text(s, f"GO1  /  LOCOMOTION LAB                                      {len(prs.slides):02}", .6, .25, 12.2, .4, 12, "22D3EE")
        text(s, title, .6, .9, 12.1, 1.1, 32, bold=True)
        if subtitle:
            text(s, subtitle, .65, 2.05, 12, .8, 18, "94A3B8")
        return s

    s = slide("Unitree Go1 보행 강화학습", "MuJoCo + PPO | CPU에서 보행 파라미터를 학습하고 보상 설계를 비교")
    text(s, "속도만 높이는 정책에서\n목표 속도·자세·에너지를 고려하는 정책으로", .7, 3.05, 7.3, 2.2, 30)
    s.shapes.add_picture(str(root / "improved.png"), Inches(8.1), Inches(3), width=Inches(4.7))
    s.notes_slide.notes_text_frame.text = "대각선 다리가 교대하는 트로트 위상은 미리 제공했습니다. PPO가 여섯 보행 파라미터를 학습하는 실험이며 end-to-end 관절 제어 학습과 구별합니다."

    s = slide("01  목표와 실행 환경")
    text(s, "목표: 평지에서 Go1의 안정적인 전진 운동\n시뮬레이터: MuJoCo 4.0.0 / Unitree Go1 MJCF\n알고리즘: Stable-Baselines3 PPO / PyTorch CPU\n하드웨어: 현재 클라우드에 GPU 없음\n검증: 물리 실행·환경 API·재현성·EGL 영상 저장", .7, 2.2, 12, 4.9)

    s = slide("02  사람이 정한 부분과 학습한 부분", "대각선 트로트 + 역기구학을 보행의 사전 지식으로 제공")
    text(s, "PPO → 보행 파라미터 6개 → 발 궤적 → 역기구학 → 관절 위치 제어", .7, 2.7, 12, 1, 25, "22D3EE", True)
    text(s, "학습: 보폭, 주파수, 다리 길이, 발 높이, 발 중심 위치, 지지 비율\n고정: 다리 위상·역기구학·약한 롤 자세 피드백\n하나의 행동 = 최대 5초 물리 에피소드 / 관측은 상수 / gamma=0\n물리 간격 4ms · 관절 목표 갱신 20ms", .7, 4, 12, 3, 21)
    s.notes_slide.notes_text_frame.text = "일반적인 매 제어 스텝의 폐루프 신경망 정책이 아니라 에피소드 단위 보행 파라미터 최적화입니다. 엄밀한 네 발 순차 walk가 아니라 trot입니다."

    s = slide("03  보상 설계 실험")
    text(s, "BASELINE", .7, 2.3, 5.6, .6, 22, "FBBF24", True)
    text(s, "전진 속도 vx 보상\n빠른 이동을 직접 장려\n목표 속도·에너지 제한 없음", .7, 3.1, 5.6, 2.5, 23)
    text(s, "IMPROVED", 6.8, 2.3, 5.8, .6, 22, "22D3EE", True)
    text(s, "목표 0.25m/s 추종\n몸통 기울기·옆 속도·회전 억제\n절대 기계적 파워 페널티", 6.8, 3.1, 5.8, 2.5, 23)
    text(s, "공통: 넘어지면 조기 종료 + 벌점 / 같은 PPO와 학습 예산", .7, 6.3, 12, .6, 19, "94A3B8")
    s.notes_slide.notes_text_frame.text = "개선 보상은 exp(-(vx-0.25)^2/0.04) -2*tilt -0.5*vy^2 -0.003*power -0.1*yaw_rate^2입니다. 전체 예정 시간으로 정규화하며 넘어지면 -2를 부여합니다. 서로 다른 보상의 수치를 직접 비교하지 않습니다."

    s = slide("04  학습 및 평가 방법")
    text(s, f"학습: 조건별 {configs['baseline']['actual_rollouts']:,}개 롤아웃, seed=42\n검증 시드 900~902의 평균 보상으로 체크포인트 선택\n평가: 별도 시드 1000~1009, 각 최대 12초\n초기 관절각에 작은 잡음 / 평지·고정 마찰\n비교: 초기 파라미터 vs baseline PPO vs improved PPO", .7, 2.3, 12, 4.5, 24)
    s.notes_slide.notes_text_frame.text = "10개 평가 시드는 독립 학습 10회가 아니라 동일 정책을 서로 다른 초기 상태에서 평가한 것입니다. 초기 상태 잡음 외의 도메인 랜덤화는 하지 않았습니다."

    s = slide("05  실제 학습 곡선", "각 그래프는 해당 조건의 자체 보상 · 두 조건의 보상 크기를 직접 비교하지 않음")
    s.shapes.add_picture(str(root / "learning_curves.png"), Inches(.7), Inches(2.9), width=Inches(11.9))

    s = slide("06  학습 전후 평가", "막대: 10개 별도 초기 상태의 평균 / 오차 막대: 표준편차")
    s.shapes.add_picture(str(root / "comparison.png"), Inches(.6), Inches(3), width=Inches(12.1))

    s = slide("07  수치 결과와 해석")
    table = s.shapes.add_table(4, 4, Inches(.7), Inches(2.4), Inches(11.9), Inches(2.4)).table
    contents = [["정책", "속도 (m/s)", "기계적 파워 (W)", "넘어짐"]]
    for name, label in [("initial", "초기"), ("baseline", "속도 중심"), ("improved", "개선 보상")]:
        r = data[name]
        contents.append([label, f"{r['speed_m_s']['mean']:.3f}", f"{r['mean_power_W']['mean']:.1f}", f"{r['falls']}/{r['episodes']}"])
    for row, values in enumerate(contents):
        for col, value in enumerate(values):
            cell = table.cell(row, col)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string("164E63" if row == 0 else "1E293B")
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name, paragraph.font.size = "Noto Sans CJK KR", Pt(22)
                paragraph.font.color.rgb = RGBColor.from_string("E2E8F0")
    text(s, f"목표 속도: 0.25m/s\n속도 추종 점수: baseline {data['baseline']['tracking_score']['mean']:.3f} → improved {data['improved']['tracking_score']['mean']:.3f}\n기계적 파워는 배터리 전력과 다름", .7, 5.3, 12, 1.8, 21, "22D3EE")

    s = slide("08  보행 영상과 한계")
    s.shapes.add_picture(str(root / "improved.png"), Inches(.7), Inches(2.4), width=Inches(6))
    text(s, "별도 영상: improved.mp4\n비교 영상: baseline.mp4\n\n한계\n단일 학습 시드 / 평지 한정\n트로트 위상은 미리 제공\n실제 로봇 성능은 검증하지 않음", 7.1, 2.4, 5.6, 4.6, 21)

    s = slide("09  다음 실험")
    text(s, "① 학습 시드 3개 이상으로 결과 재현성 확인\n② 발 접촉·미끄럼 항 추가 및 단일 항 제거 실험\n③ 마찰·질량·외력 변화를 주어 일반화 평가\n④ 위상 학습으로 walk / trot 비교\n⑤ GPU 환경에서 상태 관측 기반 residual PPO로 확장", .7, 2.4, 12, 4.5, 24)
    text(s, "자료: MuJoCo Menagerie / Unitree Go1 · Stable-Baselines3 PPO", .7, 6.8, 12, .5, 15, "94A3B8")
    s.notes_slide.notes_text_frame.text = "모델: https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/unitree_go1\n알고리즘: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html\n시뮬레이터: https://mujoco.readthedocs.io/"
    prs.save(root / "발표자료.pptx")

    report = ["# Go1 보행 실험 기록", "", "## 실제 실행한 실험", "", "MuJoCo 4.0.0 + Stable-Baselines3 PPO + CPU. 모델 및 메시는 Menagerie Go1 원본이다.", "", "PPO는 미리 정의된 트로트 궤적의 파라미터 6개를 학습한다. end-to-end 보행 학습이나 실물 로봇 검증은 수행하지 않았다.", "", "| 조건 | 실제 학습 롤아웃 | 학습 벽시계 시간(초) |", "|---|---:|---:|"]
    for name, config in configs.items():
        report.append(f"| {name} | {config['actual_rollouts']} | {config['wall_time_s']:.1f} |")
    report += ["", "## 공통 평가: 별도 초기 상태 10개 × 최대 12초", "", "| 정책 | 전방 속도 m/s | 평균 기계적 파워 W | 속도 추종 점수 | 넘어짐 |", "|---|---:|---:|---:|---:|"]
    for name, r in data.items():
        report.append(f"| {name} | {r['speed_m_s']['mean']:.4f} ± {r['speed_m_s']['std']:.4f} | {r['mean_power_W']['mean']:.2f} ± {r['mean_power_W']['std']:.2f} | {r['tracking_score']['mean']:.4f} | {r['falls']}/{r['episodes']} |")
    base, imp = data["baseline"], data["improved"]
    change = 100 * (1 - imp["mean_power_W"]["mean"] / base["mean_power_W"]["mean"])
    report += ["", "## 관측과 해석", "", f"- improved의 평균 기계적 파워는 baseline 대비 {change:.1f}% {'감소' if change >= 0 else '증가'}했다. 배터리 사용량 절감률을 뜻하지 않는다.", f"- 속도 추종 점수는 {base['tracking_score']['mean']:.4f}에서 {imp['tracking_score']['mean']:.4f}로 바뀌었다. 목표는 최고 속도가 아니라 0.25m/s 추종이다.", "- 보상에 여러 항을 동시에 추가했으므로 어떤 항 하나가 변화를 만들었다고 단정할 수 없다.", "- 평가 표준편차는 초기 상태 차이만 반영한다. 학습 시드에 따른 불확실성은 측정하지 않았다.", "- 영상은 평가 시드 1000의 실제 시뮬레이션이다. 미끄럼·접촉 품질·자연스러움의 정량 평가와 실물 검증은 후속 과제다.", "", "## 선택된 파라미터", ""]
    for name, r in data.items():
        report += [f"### {name}", "", "```json", json.dumps(r["parameters"], indent=2), "```", ""]
    report += ["## 재실행과 원자료", "", "실행 명령, 보상 수식, 한계는 ../README.md에 있다. evaluation.csv는 평가 에피소드별 원자료, 각 조건의 training.csv는 학습 롤아웃별 원자료다. 체크포인트와 config.json도 함께 보존한다."]
    (root / "실험기록.md").write_text("\n".join(report), encoding="utf-8")
    print(root / "발표자료.pptx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("go1_walk/results"))
    generate(parser.parse_args().results)
