"""Multi-window regression with an early proposal and a late reversal."""
from backend.engine import Engine
from backend.insights import Segment, extract, sentences, summarize


def run(pipeline):
    texts = ["We propose releasing the app on Friday. The schedule is tentative."]
    topics = ["support coverage", "privacy training", "test documentation", "design review"]
    for i in range(60):
        texts.append(f"Agenda item {i+1} covers {topics[i%4]}. The team reviewed the available staff and recorded open questions for the next planning session.")
    texts.append("We decided to cancel the Friday release and ship on Monday instead.")
    rows = list(sentences([Segment(start=30*i,end=30*i+10,text=t) for i,t in enumerate(texts)]))
    stages = []
    output, _ = summarize(rows, pipeline, lambda stage,*args: stages.append(stage))
    assert "synthesizing" in stages
    assert texts[-1] in output and extract(rows)[-1].start == 1830
    return output


if __name__=="__main__":
    print(run(Engine().summarizer))
