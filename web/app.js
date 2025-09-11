const { useState, useRef, useEffect } = React;

const TRACKS = ["Voice", "Music", "SFX", "Markers"];
const INITIAL_SEGMENTS = [
  { id: 1, track: "Voice", start: 0, duration: 20, speaker: "A" },
  { id: 2, track: "Music", start: 5, duration: 30 },
  { id: 3, track: "SFX", start: 15, duration: 10 },
];

function App() {
  const [segments, setSegments] = useState(INITIAL_SEGMENTS);
  const [selected, setSelected] = useState(null);
  const [zoom, setZoom] = useState(10); // pixels per unit
  const [selection, setSelection] = useState(null);

  useEffect(() => {
    const progress = new WebSocket(`ws://${location.host}/ws/progress`);
    progress.onmessage = (e) => console.log("progress", e.data);
    const preview = new WebSocket(`ws://${location.host}/ws/preview`);
    preview.onmessage = (e) => console.log("preview", e.data);
    return () => {
      progress.close();
      preview.close();
    };
  }, []);

  const handleDrag = (id, delta) => {
    setSegments((segs) =>
      segs.map((s) =>
        s.id === id ? { ...s, start: Math.max(0, Math.round((s.start + delta) / 5) * 5) } : s
      )
    );
  };

  return (
    React.createElement(
      "div",
      null,
      React.createElement("div", null,
        "Zoom:",
        React.createElement("input", {
          type: "range",
          min: 5,
          max: 40,
          value: zoom,
          onChange: (e) => setZoom(Number(e.target.value)),
        })
      ),
      React.createElement(Timeline, {
        segments,
        zoom,
        selection,
        setSelection,
        onDrag: handleDrag,
        onSelect: setSelected,
      }),
      React.createElement(Inspector, { segment: segments.find((s) => s.id === selected) })
    )
  );
}

function Timeline({ segments, zoom, selection, setSelection, onDrag, onSelect }) {
  const ref = useRef(null);
  const [selStart, setSelStart] = useState(null);

  const onMouseDown = (e) => {
    if (e.target === ref.current) {
      const rect = ref.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setSelStart(x);
      setSelection({ start: x, end: x });
    }
  };

  const onMouseMove = (e) => {
    if (selStart !== null) {
      const rect = ref.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setSelection({ start: selStart, end: x });
    }
  };

  const onMouseUp = () => {
    setSelStart(null);
  };

  return React.createElement(
    "div",
    {
      id: "timeline",
      ref,
      onMouseDown,
      onMouseMove,
      onMouseUp,
    },
    selection &&
      React.createElement("div", {
        className: "selection",
        style: {
          left: Math.min(selection.start, selection.end) + "px",
          width: Math.abs(selection.end - selection.start) + "px",
        },
      }),
    TRACKS.map((t) =>
      React.createElement(Track, {
        key: t,
        name: t,
        segments: segments.filter((s) => s.track === t),
        zoom,
        onDrag,
        onSelect,
      })
    )
  );
}

function Track({ name, segments, zoom, onDrag, onSelect }) {
  return React.createElement(
    "div",
    { className: "track" },
    segments.map((s) =>
      React.createElement(Segment, {
        key: s.id,
        segment: s,
        zoom,
        onDrag,
        onSelect,
      })
    )
  );
}

function Segment({ segment, zoom, onDrag, onSelect }) {
  const ref = useRef(null);
  const [dragStart, setDragStart] = useState(null);

  const onMouseDown = (e) => {
    setDragStart(e.clientX);
    onSelect(segment.id);
    e.stopPropagation();
  };

  const onMouseMove = (e) => {
    if (dragStart !== null) {
      const dx = (e.clientX - dragStart) / zoom;
      setDragStart(e.clientX);
      onDrag(segment.id, dx);
    }
  };

  const onMouseUp = () => setDragStart(null);

  useEffect(() => {
    const el = ref.current;
    el.addEventListener("mousemove", onMouseMove);
    el.addEventListener("mouseup", onMouseUp);
    return () => {
      el.removeEventListener("mousemove", onMouseMove);
      el.removeEventListener("mouseup", onMouseUp);
    };
  });

  useEffect(() => {
    const canvas = ref.current.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x += 4) {
      const y = Math.random() * canvas.height;
      ctx.moveTo(x, canvas.height / 2);
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#007bff";
    ctx.stroke();
  }, [segment.start, segment.duration, zoom]);

  const laneIndex = segment.speaker ? segment.speaker.charCodeAt(0) % 3 : 0;

  return React.createElement(
    "div",
    {
      ref,
      className: "segment",
      onMouseDown,
      style: {
        left: segment.start * zoom + "px",
        width: segment.duration * zoom + "px",
        top: 10 + laneIndex * 20 + "px",
      },
    },
    React.createElement("canvas", { width: segment.duration * zoom, height: 40 })
  );
}

function Inspector({ segment }) {
  if (!segment) {
    return React.createElement("div", { id: "inspector" }, "Select a segment");
  }
  return React.createElement(
    "div",
    { id: "inspector" },
    React.createElement("h4", null, "Inspector"),
    React.createElement("div", null, `Track: ${segment.track}`),
    React.createElement("div", null, `Start: ${segment.start}`),
    React.createElement("div", null, `Duration: ${segment.duration}`),
    segment.speaker && React.createElement("div", null, `Speaker: ${segment.speaker}`)
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
