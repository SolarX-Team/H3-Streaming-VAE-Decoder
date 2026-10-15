/* Citation previews: hovering or focusing [n] shows the reference without leaving the paragraph.
   Clicking still jumps to the list at the end. */
(() => {
  "use strict";
  const pop = document.createElement("div");
  pop.className = "cite-pop";
  pop.setAttribute("role", "tooltip");
  document.body.appendChild(pop);
  let current = null, hideTimer = 0;

  function place(a) {
    const r = a.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const x = Math.max(12, Math.min(r.left + r.width / 2 - pop.offsetWidth / 2, vw - pop.offsetWidth - 12));
    let y = r.bottom + 8;
    if (y + pop.offsetHeight > innerHeight - 12) y = r.top - pop.offsetHeight - 8;
    pop.style.left = `${x + scrollX}px`;
    pop.style.top = `${y + scrollY}px`;
  }

  function open(a) {
    const item = document.getElementById(a.hash.slice(1));
    if (!item) return;
    clearTimeout(hideTimer);
    if (current !== a) {
      current = a;
      pop.innerHTML = `<span class="n">${item.querySelector(".n").textContent}</span> ${item.querySelector(".body").innerHTML}`;
    }
    pop.classList.add("show");
    place(a);
  }

  function close(now) {
    clearTimeout(hideTimer);
    const hide = () => { pop.classList.remove("show"); current = null; };
    if (now) hide();
    else hideTimer = setTimeout(hide, 150);
  }

  document.addEventListener("mouseover", e => {
    const a = e.target.closest(".cite a");
    if (a) open(a);
    else if (e.target.closest(".cite-pop")) clearTimeout(hideTimer);
  });
  document.addEventListener("mouseout", e => {
    if (e.target.closest(".cite a, .cite-pop")) close();
  });
  document.addEventListener("focusin", e => {
    const a = e.target.closest(".cite a");
    if (a) open(a);
  });
  document.addEventListener("focusout", e => {
    if (e.target.closest(".cite a")) close();
  });
  document.addEventListener("click", e => {
    if (e.target.closest(".cite a")) close(true);
  });
  addEventListener("scroll", () => { if (current) close(true); }, { passive: true });
})();
