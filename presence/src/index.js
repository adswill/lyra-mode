import { DurableObject } from "cloudflare:workers";

const ONLINE_MS = 90 * 1000;

function utcDay(ms = Date.now()) {
  return new Date(ms).toISOString().slice(0, 10);
}

function dayStamp(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return null;
  }
  return value;
}

export class Presence extends DurableObject {
  async heartbeat(id, user) {
    const now = Date.now();
    const day = utcDay(now);
    await this.ctx.storage.put("s:" + id, now);
    if (user) {
      await this.ctx.storage.put("u:" + day + ":" + user, 1);
    }
    return this.count(now);
  }

  async leave(id) {
    await this.ctx.storage.delete("s:" + id);
    return this.count(Date.now());
  }

  async count(now = Date.now()) {
    const cutoff = now - ONLINE_MS;
    const map = await this.ctx.storage.list({ prefix: "s:" });
    let n = 0;
    const stale = [];
    for (const [key, seen] of map) {
      if (typeof seen !== "number" || seen < cutoff) {
        stale.push(key);
      } else {
        n += 1;
      }
    }
    if (stale.length) {
      await this.ctx.storage.delete(stale);
    }
    return n;
  }

  async unique(day) {
    const map = await this.ctx.storage.list({ prefix: "u:" + day + ":" });
    return map.size;
  }
}

function json(body, status = 200) {
  const res = new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
  res.headers.set("access-control-allow-origin", "*");
  res.headers.set("access-control-allow-methods", "GET,POST,OPTIONS");
  res.headers.set("access-control-allow-headers", "content-type");
  return res;
}

function asUuid(value) {
  const id = typeof value === "string" ? value.trim() : "";
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    return null;
  }
  return id.toLowerCase();
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return json({}, 204);
    }
    const url = new URL(request.url);
    const stub = env.PRESENCE.get(env.PRESENCE.idFromName("global"));
    try {
      if (url.pathname === "/v1/heartbeat" && request.method === "POST") {
        const body = await request.json();
        const id = asUuid(body?.id);
        const user = asUuid(body?.user);
        if (!id) {
          return json({ error: "bad id" }, 400);
        }
        const online = await stub.heartbeat(id, user);
        console.log(JSON.stringify({ event: "heartbeat", online }));
        return json({ online });
      }
      if (url.pathname === "/v1/leave" && request.method === "POST") {
        const id = asUuid((await request.json())?.id);
        if (!id) {
          return json({ error: "bad id" }, 400);
        }
        const online = await stub.leave(id);
        return json({ online });
      }
      if (url.pathname === "/v1/count" && request.method === "GET") {
        const online = await stub.count();
        return json({ online });
      }
      if (url.pathname === "/v1/unique" && request.method === "GET") {
        const day = dayStamp(url.searchParams.get("day") || "") || utcDay();
        const unique = await stub.unique(day);
        return json({ day, unique });
      }
      return json({ error: "not found" }, 404);
    } catch (err) {
      console.log(JSON.stringify({ event: "error", err: String(err) }));
      return json({ error: "server" }, 500);
    }
  },
};
