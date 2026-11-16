import { useEffect, useState } from "react";
import ProjectList from "./components/ProjectList";
import ProjectPage from "./components/ProjectPage";

function useHash() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHash();
  const m = hash.match(/^#\/p\/([0-9a-f]{32})/);
  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand">
          Milestone Photo Normalizer
        </a>
      </header>
      {m ? <ProjectPage key={m[1]} id={m[1]} /> : <ProjectList />}
    </div>
  );
}
