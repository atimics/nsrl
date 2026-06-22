#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = {
  outDir: "data/processed/cosyworld-kernel-corpus",
  cosyworldRoot: "/Users/ratimics/develop/cosyworld",
  seed: "cosyworld-kernel-v1",
  cc: process.env.CC || "cc",
};

const actorNames = new Map([
  [1001, "Rati"],
  [1002, "Whiskerwind"],
  [1003, "Skull"],
  [1004, "Moonlit Echo"],
  [1005, "Old Oak"],
  [5001, "Visitor"],
]);

const locationNames = new Map([
  [1, "Cottage Hearth"],
  [2, "Moss Garden"],
  [3, "Training Trail"],
  [10, "Archive Loft"],
  [11, "Greenhouse Hall"],
  [12, "Pantry Nook"],
  [13, "Teacup Observatory"],
  [14, "Workshop Alcove"],
  [15, "Moonlit Bridge"],
  [40, "Old Oak Hollow"],
]);

const itemNames = new Map([
  [2001, "Hearth Tonic"],
  [2002, "Moss Charm"],
  [2003, "Moon Charm"],
  [2004, "Archive Charm"],
  [2005, "Hearth Charm"],
  [2006, "Trail Charm"],
  [2007, "Garden Charm"],
]);

const itemKinds = new Map([
  [1, "potion"],
  [2, "evolution charm"],
]);

const abilityNames = new Map([
  [0, "strength"],
  [1, "dexterity"],
  [2, "constitution"],
  [3, "intelligence"],
  [4, "wisdom"],
  [5, "charisma"],
]);

const offerFlagNames = [
  [1 << 0, "chat"],
  [1 << 1, "check"],
  [1 << 2, "pick up"],
  [1 << 3, "use item"],
  [1 << 4, "defend"],
  [1 << 5, "attack"],
  [1 << 6, "move"],
  [1 << 7, "give item"],
  [1 << 8, "flee"],
];

const probeSource = String.raw`
#include "cosy_kernel.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *status_name(cw_status status) {
  switch (status) {
    case CW_OK: return "ok";
    case CW_ERR_INVALID: return "invalid";
    case CW_ERR_FULL: return "full";
    case CW_ERR_NOT_FOUND: return "not_found";
    case CW_ERR_RULE: return "rule";
    default: return "unknown";
  }
}

static const char *action_name(uint8_t kind) {
  switch (kind) {
    case CW_ACTION_CREATE_ACTOR: return "create_actor";
    case CW_ACTION_SAY: return "say";
    case CW_ACTION_MOVE: return "move";
    case CW_ACTION_ABILITY_CHECK: return "ability_check";
    case CW_ACTION_PICK_UP_ITEM: return "pick_up_item";
    case CW_ACTION_USE_ITEM: return "use_item";
    case CW_ACTION_ATTACK: return "attack";
    case CW_ACTION_DEFEND: return "defend";
    case CW_ACTION_GIVE_ITEM: return "give_item";
    case CW_ACTION_FLEE: return "flee";
    default: return "none";
  }
}

static void print_event(const cw_event *event) {
  printf("{\"seq\":%" PRIu64 ",\"type\":%u,\"type_name\":\"%s\","
         "\"success\":%u,\"reason\":%u,"
         "\"actor_id\":%" PRIu64 ",\"target_actor_id\":%" PRIu64 ","
         "\"location_id\":%" PRIu64 ",\"destination_location_id\":%" PRIu64 ","
         "\"content_id\":%" PRIu64 ",\"item_id\":%" PRIu64 ","
         "\"raw_roll\":%d,\"modifier\":%d,\"total\":%d,\"dc\":%d,"
         "\"damage\":%d,\"current_hp\":%d}",
         event->seq,
         (unsigned)event->type,
         cw_event_type_name(event->type),
         (unsigned)event->success,
         (unsigned)event->reason,
         event->actor_id,
         event->target_actor_id,
         event->location_id,
         event->destination_location_id,
         event->content_id,
         event->item_id,
         event->raw_roll,
         event->modifier,
         event->total,
         event->dc,
         event->damage,
         event->current_hp);
}

static void print_actor(const cw_actor *actor) {
  printf("{\"id\":%" PRIu64 ",\"kind\":%u,\"status\":%u,"
         "\"location_id\":%" PRIu64 ",\"hp_base\":%d,\"current_hp\":%d,"
         "\"damage\":%d,\"level\":%u,\"conditions\":%u,"
         "\"str\":%d,\"dex\":%d,\"con\":%d,\"int\":%d,\"wis\":%d,\"cha\":%d}",
         actor->id,
         (unsigned)actor->kind,
         (unsigned)actor->status,
         actor->location_id,
         actor->stats.hp_base,
         cw_actor_current_hp(actor),
         actor->damage,
         (unsigned)actor->stats.level,
         (unsigned)actor->conditions,
         actor->stats.strength,
         actor->stats.dexterity,
         actor->stats.constitution,
         actor->stats.intelligence,
         actor->stats.wisdom,
         actor->stats.charisma);
}

static void print_item(const cw_item *item) {
  printf("{\"id\":%" PRIu64 ",\"kind\":%u,\"charges\":%u,"
         "\"location_id\":%" PRIu64 ",\"holder_actor_id\":%" PRIu64 ","
         "\"recharge_at_tick\":%" PRIu64 "}",
         item->id,
         (unsigned)item->kind,
         (unsigned)item->charges,
         item->location_id,
         item->holder_actor_id,
         item->recharge_at_tick);
}

static void print_location(const cw_location *location) {
  printf("{\"id\":%" PRIu64 ",\"flags\":%u}",
         location->id,
         (unsigned)location->flags);
}

static void print_exit(const cw_exit *exit) {
  printf("{\"from_location_id\":%" PRIu64 ",\"to_location_id\":%" PRIu64 ",\"flags\":%u}",
         exit->from_location_id,
         exit->to_location_id,
         (unsigned)exit->flags);
}

static void print_offers(const cw_world *world, const cw_actor *actor) {
  cw_action_offers offers;
  memset(&offers, 0, sizeof(offers));
  cw_status status = cw_get_action_offers(world, actor->id, &offers);
  printf("{\"actor_id\":%" PRIu64 ",\"status\":\"%s\",\"option_flags\":%u}",
         actor->id,
         status_name(status),
         status == CW_OK ? offers.option_flags : 0u);
}

static void emit_snapshot(const char *stage,
                          int step,
                          const char *action,
                          uint64_t seed,
                          cw_status status,
                          const cw_action *applied,
                          const cw_world *world,
                          const cw_event_buffer *events) {
  printf("{\"schema\":\"nsrl.cosyworld_kernel_state.v1\","
         "\"stage\":\"%s\",\"step\":%d,\"action\":\"%s\","
         "\"seed\":%" PRIu64 ",\"status\":\"%s\","
         "\"tick\":%" PRIu64 ",\"next_event_seq\":%" PRIu64 ","
         "\"actor_count\":%zu,\"item_count\":%zu,\"location_count\":%zu,\"exit_count\":%zu",
         stage,
         step,
         action,
         seed,
         status_name(status),
         world->tick,
         world->next_event_seq,
         world->actor_count,
         world->item_count,
         world->location_count,
         world->exit_count);

  if (applied) {
    printf(",\"applied\":{\"kind\":%u,\"kind_name\":\"%s\","
           "\"ability\":%u,\"dc\":%u,"
           "\"actor_id\":%" PRIu64 ",\"target_actor_id\":%" PRIu64 ","
           "\"location_id\":%" PRIu64 ",\"destination_location_id\":%" PRIu64 ","
           "\"content_id\":%" PRIu64 ",\"item_id\":%" PRIu64 "}",
           (unsigned)applied->kind,
           action_name(applied->kind),
           (unsigned)applied->ability,
           (unsigned)applied->dc,
           applied->actor_id,
           applied->target_actor_id,
           applied->location_id,
           applied->destination_location_id,
           applied->content_id,
           applied->item_id);
  } else {
    printf(",\"applied\":null");
  }

  printf(",\"events\":[");
  if (events) {
    for (size_t i = 0; i < events->count; ++i) {
      if (i) printf(",");
      print_event(&events->events[i]);
    }
  }
  printf("],\"actors\":[");
  for (size_t i = 0; i < world->actor_count; ++i) {
    if (i) printf(",");
    print_actor(&world->actors[i]);
  }
  printf("],\"items\":[");
  for (size_t i = 0; i < world->item_count; ++i) {
    if (i) printf(",");
    print_item(&world->items[i]);
  }
  printf("],\"locations\":[");
  for (size_t i = 0; i < world->location_count; ++i) {
    if (i) printf(",");
    print_location(&world->locations[i]);
  }
  printf("],\"exits\":[");
  for (size_t i = 0; i < world->exit_count; ++i) {
    if (i) printf(",");
    print_exit(&world->exits[i]);
  }
  printf("],\"offers\":[");
  for (size_t i = 0; i < world->actor_count; ++i) {
    if (i) printf(",");
    print_offers(world, &world->actors[i]);
  }
  printf("]}\n");
}

static cw_actor *find_actor_local(cw_world *world, cw_id id) {
  for (size_t i = 0; i < world->actor_count; ++i) {
    if (world->actors[i].id == id) return &world->actors[i];
  }
  return 0;
}

static void apply_and_emit(cw_world *world,
                           int step,
                           const char *stage,
                           cw_action action,
                           uint64_t seed) {
  cw_event_buffer events;
  cw_status status = cw_world_apply(world, &action, seed, &events);
  emit_snapshot(stage, step, action_name(action.kind), seed, status, &action, world, &events);
}

int main(void) {
  cw_world world;
  cw_event_buffer events;
  cw_world_init(&world);
  cw_status status = cw_seed_cosy_cottage(&world, &events);
  emit_snapshot("seed", 0, "seed_cosy_cottage", 0, status, 0, &world, &events);

  cw_action action;
  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_CREATE_ACTOR;
  action.actor_id = 5001;
  action.location_id = 1;
  apply_and_emit(&world, 1, "visitor_enters", action, 42);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_SAY;
  action.actor_id = 5001;
  action.content_id = 9001;
  apply_and_emit(&world, 2, "visitor_speaks", action, 42);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 1001;
  action.destination_location_id = 3;
  apply_and_emit(&world, 3, "rati_tests_locked_path", action, 99);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 1001;
  action.destination_location_id = 2;
  apply_and_emit(&world, 4, "rati_walks_to_garden", action, 100);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_ABILITY_CHECK;
  action.actor_id = 1001;
  action.ability = CW_ABILITY_WISDOM;
  action.dc = 12;
  apply_and_emit(&world, 5, "rati_checks_garden", action, 1234);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_PICK_UP_ITEM;
  action.actor_id = 1001;
  action.item_id = 2007;
  apply_and_emit(&world, 6, "rati_collects_garden_charm", action, 55);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 1001;
  action.destination_location_id = 1;
  apply_and_emit(&world, 7, "rati_returns_hearth", action, 101);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_PICK_UP_ITEM;
  action.actor_id = 1001;
  action.item_id = 2001;
  apply_and_emit(&world, 8, "rati_takes_tonic", action, 56);

  cw_actor *rati = find_actor_local(&world, 1001);
  if (rati) rati->damage = 5;
  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_USE_ITEM;
  action.actor_id = 1001;
  action.target_actor_id = 1001;
  action.item_id = 2001;
  apply_and_emit(&world, 9, "rati_uses_tonic", action, 57);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 1003;
  action.destination_location_id = 2;
  apply_and_emit(&world, 10, "skull_moves_garden", action, 58);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 1003;
  action.destination_location_id = 3;
  apply_and_emit(&world, 11, "skull_moves_trail", action, 59);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_DEFEND;
  action.actor_id = 1003;
  apply_and_emit(&world, 12, "skull_defends", action, 60);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_ATTACK;
  action.actor_id = 1003;
  action.target_actor_id = 1004;
  apply_and_emit(&world, 13, "skull_spars_echo", action, 61);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_FLEE;
  action.actor_id = 1003;
  action.destination_location_id = 2;
  apply_and_emit(&world, 14, "skull_leaves_trail", action, 62);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 5001;
  action.destination_location_id = 2;
  apply_and_emit(&world, 15, "visitor_walks_garden", action, 43);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_PICK_UP_ITEM;
  action.actor_id = 5001;
  action.item_id = 2002;
  apply_and_emit(&world, 16, "visitor_collects_moss_charm", action, 44);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 5001;
  action.destination_location_id = 3;
  apply_and_emit(&world, 17, "visitor_walks_trail", action, 45);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_PICK_UP_ITEM;
  action.actor_id = 5001;
  action.item_id = 2003;
  apply_and_emit(&world, 18, "visitor_collects_moon_charm", action, 46);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 5001;
  action.destination_location_id = 2;
  apply_and_emit(&world, 19, "visitor_returns_garden", action, 47);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_MOVE;
  action.actor_id = 5001;
  action.destination_location_id = 1;
  apply_and_emit(&world, 20, "visitor_returns_hearth", action, 48);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_GIVE_ITEM;
  action.actor_id = 5001;
  action.target_actor_id = 1002;
  action.item_id = 2002;
  apply_and_emit(&world, 21, "visitor_gives_moss_charm", action, 49);

  memset(&action, 0, sizeof(action));
  action.kind = CW_ACTION_GIVE_ITEM;
  action.actor_id = 5001;
  action.target_actor_id = 1002;
  action.item_id = 2003;
  apply_and_emit(&world, 22, "visitor_gives_moon_charm", action, 50);

  return 0;
}
`;

function usage() {
  console.log(`Usage: node scripts/build-cosyworld-kernel-corpus.mjs [options]

Options:
  --out-dir PATH          Output directory [${defaults.outDir}]
  --cosyworld-root PATH   CosyWorld checkout root [${defaults.cosyworldRoot}]
  --seed TEXT             Stable id seed [${defaults.seed}]
  --cc PATH               C compiler [${defaults.cc}]
`);
}

function parseArgs(argv) {
  const options = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in options)) {
      throw new Error(`unknown option: ${arg}`);
    }
    const value = argv[++index];
    if (value === undefined) {
      throw new Error(`${arg} requires a value`);
    }
    options[key] = value;
  }
  return options;
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function cleanAscii(text) {
  return String(text ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    encoding: "utf8",
    stdio: options.capture ? ["ignore", "pipe", "pipe"] : ["ignore", "inherit", "pipe"],
  });
  if (result.status !== 0) {
    const stderr = result.stderr ? `\n${result.stderr.trim()}` : "";
    throw new Error(`${command} failed with status ${result.status}${stderr}`);
  }
  return result;
}

function compileProbe(options, outDir) {
  const coreRoot = path.join(options.cosyworldRoot, "v2/core-c");
  const includeDir = path.join(coreRoot, "include");
  const kernelSource = path.join(coreRoot, "src/cosy_kernel.c");
  if (!fs.existsSync(path.join(includeDir, "cosy_kernel.h"))) {
    throw new Error(`missing CosyWorld kernel header under ${includeDir}`);
  }
  if (!fs.existsSync(kernelSource)) {
    throw new Error(`missing CosyWorld kernel source: ${kernelSource}`);
  }

  const buildDir = path.join(outDir, "_build");
  fs.mkdirSync(buildDir, { recursive: true });
  const probePath = path.join(buildDir, "cosyworld-kernel-probe.c");
  const binaryPath = path.join(buildDir, "cosyworld-kernel-probe");
  fs.writeFileSync(probePath, probeSource, "utf8");

  run(options.cc, [
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-O2",
    `-I${includeDir}`,
    probePath,
    kernelSource,
    "-o",
    binaryPath,
  ]);
  return { probePath, binaryPath, kernelSource, includeDir };
}

function parseJsonl(text, label) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${label}:${index + 1}: ${error.message}`);
      }
    });
}

function nameFrom(map, id, fallback) {
  if (!id) return "";
  return map.get(Number(id)) || `${fallback} ${id}`;
}

function actorName(id) {
  return nameFrom(actorNames, id, "Actor");
}

function locationName(id) {
  return nameFrom(locationNames, id, "Location");
}

function itemName(id) {
  return nameFrom(itemNames, id, "Item");
}

function listWords(values, empty = "none") {
  const clean = values.map(cleanAscii).filter(Boolean);
  if (clean.length === 0) return empty;
  if (clean.length === 1) return clean[0];
  if (clean.length === 2) return `${clean[0]} and ${clean[1]}`;
  return `${clean.slice(0, -1).join(", ")}, and ${clean.at(-1)}`;
}

function actorsAt(row, locationId) {
  return row.actors
    .filter((actor) => Number(actor.location_id) === Number(locationId) && Number(actor.status) === 1)
    .map((actor) => actorName(actor.id));
}

function itemsAt(row, locationId) {
  return row.items
    .filter((item) => Number(item.location_id) === Number(locationId) && !Number(item.holder_actor_id))
    .map((item) => itemName(item.id));
}

function exitsFrom(row, locationId) {
  return row.exits
    .filter((exit) => Number(exit.from_location_id) === Number(locationId) && !Number(exit.flags))
    .map((exit) => locationName(exit.to_location_id));
}

function actorById(row, id) {
  return row.actors.find((actor) => Number(actor.id) === Number(id));
}

function offersFor(row, actorId) {
  const offer = row.offers.find((entry) => Number(entry.actor_id) === Number(actorId));
  const flags = offer ? Number(offer.option_flags) : 0;
  return offerFlagNames
    .filter(([flag]) => flags & flag)
    .map(([, name]) => name);
}

function focusLocation(row, event) {
  const actor = actorById(row, event.actor_id);
  if (event.type_name === "move.blocked") {
    return Number(event.location_id || actor?.location_id || 1);
  }
  if (event.type_name === "actor.moved" || event.type_name === "combat.flee.success") {
    return Number(event.destination_location_id || actor?.location_id || event.location_id || 1);
  }
  return Number(actor?.location_id || event.location_id || event.destination_location_id || 1);
}

function eventMemory(row, event) {
  const actor = actorName(event.actor_id);
  const target = actorName(event.target_actor_id);
  const loc = locationName(event.location_id || focusLocation(row, event));
  const dest = locationName(event.destination_location_id);
  const item = itemName(event.item_id);
  const applied = row.applied || {};
  const ability = abilityNames.get(Number(applied.ability)) || "skill";
  const total = Number(event.total);
  const dc = Number(event.dc || applied.dc || 0);
  const success = Number(event.success) ? "success" : "failure";

  switch (event.type_name) {
    case "world.bootstrapped":
      return "The cottage has just opened its rooms, paths, residents, and pocket charms.";
    case "actor.created":
      return `${actor} arrives with fresh stats and enough health to stand in the room.`;
    case "actor.entered_location":
      return `${actor} has crossed into ${loc} and needs the next line to belong to that place.`;
    case "message.created":
      return `${actor} has spoken content ${event.content_id} at ${loc}; the room keeps the reply small and grounded.`;
    case "move.blocked":
      return `${actor} tried to reach ${dest} from ${loc}, but the cottage has no open path that way.`;
    case "actor.moved":
      return `${actor} has moved from ${loc} to ${dest}; the new threshold changes what is reachable.`;
    case "ability_check.rolled":
      return `${actor} tests ${ability}: roll ${event.raw_roll} plus ${event.modifier} gives ${total} against ${dc}, a ${success}.`;
    case "item.picked_up":
      return `${actor} now carries ${item}; the local item list and the held item list have changed.`;
    case "item.used":
      return `${actor} spends ${item}; current health settles at ${event.current_hp}.`;
    case "combat.defend":
      return `${actor} steadies their stance at ${loc} and waits behind a guarded breath.`;
    case "combat.attack.attempt":
      return `${actor} turns toward ${target} at ${loc}; the strike is committed before the result is known.`;
    case "combat.attack.hit":
      return `${actor} hits ${target} for ${event.damage}; ${target} has ${event.current_hp} health left.`;
    case "combat.attack.miss":
      return `${actor} misses ${target}; the sparring space stays tense but unbroken.`;
    case "combat.knockout":
      return `${target} drops out of the exchange at ${loc}; everyone feels the room go quiet.`;
    case "combat.flee.success":
      return `${actor} leaves combat at ${loc} and reaches ${dest}.`;
    case "item.given":
      return `${actor} gives ${item} to ${target}; the held charm now belongs with ${target}.`;
    case "avatar.evolved":
      return `${target} gathers enough matching charms to rise to level ${event.total}.`;
    case "rule.rejected":
      return `${actor || "Someone"} reaches for a rule the cottage refuses; reason ${event.reason} keeps the state intact.`;
    default:
      return `${actor || "The room"} changes state through ${event.type_name}.`;
  }
}

function expectedLine(row, event) {
  const actor = actorName(event.actor_id);
  const target = actorName(event.target_actor_id);
  const loc = locationName(event.location_id || focusLocation(row, event));
  const dest = locationName(event.destination_location_id);
  const item = itemName(event.item_id);
  const applied = row.applied || {};
  const ability = abilityNames.get(Number(applied.ability)) || "skill";
  const total = Number(event.total);
  const dc = Number(event.dc || applied.dc || 0);

  switch (event.type_name) {
    case "world.bootstrapped":
      return "Cottage Hearth wakes with familiar friends, pocket charms, and open garden paths.";
    case "actor.created":
      return `${actor} steps softly into ${loc}, carrying the hush of a first arrival.`;
    case "actor.entered_location":
      return `${actor} settles into ${loc} and listens for the room's next kindness.`;
    case "message.created":
      return `${actor} keeps the hail warm and brief beside the hearth.`;
    case "move.blocked":
      return `${actor} pauses; ${dest} is not reached from ${loc} by that path.`;
    case "actor.moved":
      return `${actor} walks from ${loc} to ${dest}, leaving the old room tidier than before.`;
    case "ability_check.rolled":
      return total >= dc
        ? `${actor} follows a clear ${ability} hunch and finds the next small truth.`
        : `${actor} studies the ${ability} sign, then marks it for another try.`;
    case "item.picked_up":
      return `${actor} pockets ${item}, which gives a quiet practical gleam.`;
    case "item.used":
      return `${actor} uses ${item} and breathes easier at ${event.current_hp} health.`;
    case "combat.defend":
      return `${actor} braces at ${loc}, careful rather than cruel.`;
    case "combat.attack.attempt":
      return `${actor} starts a measured spar with ${target} at ${loc}.`;
    case "combat.attack.hit":
      return `${actor}'s strike lands; ${target} stays at ${event.current_hp} health.`;
    case "combat.attack.miss":
      return `${actor}'s swing slips wide and the trail dust settles again.`;
    case "combat.knockout":
      return `${target} sinks out of the sparring rhythm, and the trail lowers its voice.`;
    case "combat.flee.success":
      return `${actor} backs away from ${loc} and reaches ${dest} with no flourish.`;
    case "item.given":
      return `${actor} places ${item} in ${target}'s hands with ordinary care.`;
    case "avatar.evolved":
      return `${target} brightens into level ${event.total}, steady and more themselves.`;
    case "rule.rejected":
      return `${actor || "Someone"} stops at the rule's edge and chooses a gentler next move.`;
    default:
      return `${loc} changes by one small turn.`;
  }
}

function privateState(row, event) {
  const actorId = Number(event.actor_id || row.applied?.actor_id || 1001);
  const actor = actorName(actorId);
  const locId = focusLocation(row, event);
  const loc = locationName(locId);
  const actorState = actorById(row, actorId);
  const presentActors = actorsAt(row, locId);
  const localItems = itemsAt(row, locId);
  const exits = exitsFrom(row, locId);
  const offerList = offersFor(row, actorId);
  const hpText = actorState ? `${actorState.current_hp}/${actorState.hp_base} health` : "unknown health";
  const levelText = actorState ? `level ${actorState.level}` : "unleveled";
  return cleanAscii([
    `${actor} is at ${loc} with ${hpText} and ${levelText}.`,
    `${loc} holds ${listWords(presentActors)}.`,
    `Loose items here: ${listWords(localItems)}.`,
    `Open paths: ${listWords(exits)}.`,
    `${actor} can ${listWords(offerList, "wait")}.`,
    eventMemory(row, event),
  ].join(" "));
}

function makeFrame(options, row, event, eventIndex) {
  const output = cleanAscii(expectedLine(row, event));
  const state = privateState(row, event);
  const focusActor = Number(event.actor_id || row.applied?.actor_id || 0);
  const locId = focusLocation(row, event);
  const id = idFor([options.seed, row.step, event.seq, eventIndex, state, output]);
  return {
    id,
    source: "cosyworld_c_kernel",
    source_id: `step${row.step}:event${event.seq || eventIndex}`,
    domain: "cosyworld",
    schema: "nsrl.cosyworld_kernel_frame.v1",
    stage: row.stage,
    step: row.step,
    action: row.action,
    status: row.status,
    event_type: event.type_name,
    speaker: focusActor ? actorName(focusActor) : "COSYWORLD",
    kind: "kernel_event",
    private_state: state,
    expected_output: output,
    line: output,
    target: output,
    prompt: `STATE: ${state}\nVOICE: `,
    fields: {
      actor: focusActor ? actorName(focusActor) : "",
      target_actor: actorName(event.target_actor_id),
      location: locationName(locId),
      destination: locationName(event.destination_location_id),
      item: itemName(event.item_id),
      item_kind: itemKinds.get(Number(row.items.find((item) => Number(item.id) === Number(event.item_id))?.kind)) || "",
      result: Number(event.success) ? "success" : "failure",
      reason: String(event.reason || ""),
      tick: String(row.tick),
    },
    grounding_terms: [
      focusActor ? actorName(focusActor) : "",
      actorName(event.target_actor_id),
      locationName(locId),
      locationName(event.destination_location_id),
      itemName(event.item_id),
      row.stage,
      row.action,
      event.type_name,
    ].filter(Boolean),
  };
}

function buildFrames(options, states) {
  const frames = [];
  for (const row of states) {
    const events = row.events.length ? row.events : [{
      seq: 0,
      type_name: row.action,
      actor_id: row.applied?.actor_id || 0,
      target_actor_id: row.applied?.target_actor_id || 0,
      location_id: row.applied?.location_id || 0,
      destination_location_id: row.applied?.destination_location_id || 0,
      item_id: row.applied?.item_id || 0,
      content_id: row.applied?.content_id || 0,
      success: row.status === "ok" ? 1 : 0,
      reason: 0,
      total: 0,
      dc: 0,
    }];
    for (const [index, event] of events.entries()) {
      frames.push(makeFrame(options, row, event, index));
    }
  }
  return frames;
}

function writeJsonl(filePath, records) {
  fs.writeFileSync(filePath, records.map((record) => `${JSON.stringify(record)}\n`).join(""), "utf8");
}

function trainingText(frame) {
  return `${frame.private_state}\n${frame.expected_output}\nEND\n`;
}

function trainingPair(frame) {
  return {
    id: frame.id,
    speaker: frame.speaker,
    kind: frame.kind,
    domain: frame.domain,
    event_type: frame.event_type,
    private_state: frame.private_state,
    expected_output: frame.expected_output,
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const build = compileProbe(options, outDir);
  const probe = run(build.binaryPath, [], { capture: true });
  const states = parseJsonl(probe.stdout, build.binaryPath);
  const frames = buildFrames(options, states);

  const statesPath = path.join(outDir, "states.jsonl");
  const framesPath = path.join(outDir, "frames.jsonl");
  const trainingPairsPath = path.join(outDir, "training-pairs.jsonl");
  const corpusPath = path.join(outDir, "corpus.txt");
  const voicePath = path.join(outDir, "voice.txt");
  const manifestPath = path.join(outDir, "manifest.json");

  writeJsonl(statesPath, states);
  writeJsonl(framesPath, frames);
  writeJsonl(trainingPairsPath, frames.map(trainingPair));
  fs.writeFileSync(corpusPath, `COSYWORLD_KERNEL_CORPUS_V1\n\n${frames.map(trainingText).join("\n")}`, "utf8");
  fs.writeFileSync(voicePath, `${frames.map((frame) => frame.expected_output).join("\n")}\n`, "utf8");
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.cosyworld_kernel_corpus.v1",
    created_at: new Date().toISOString(),
    cosyworld_root: options.cosyworldRoot,
    kernel_source: build.kernelSource,
    kernel_include_dir: build.includeDir,
    probe_source_path: build.probePath,
    probe_binary_path: build.binaryPath,
    states_path: statesPath,
    frames_path: framesPath,
    training_pairs_path: trainingPairsPath,
    corpus_path: corpusPath,
    voice_path: voicePath,
    state_rows: states.length,
    frames: frames.length,
    notes: [
      "Generated by compiling and running the CosyWorld C kernel from v2/core-c.",
      "Rows preserve raw numeric simulator state in states.jsonl, then convert each emitted event into private_state and expected_output.",
      "The flattened corpus is private_state newline expected_output; no RANKED wrapper is needed for the focused pair trainer.",
    ],
  }, null, 2)}\n`, "utf8");

  console.log(`states=${states.length}`);
  console.log(`frames=${frames.length}`);
  console.log(`states_path=${statesPath}`);
  console.log(`frames_path=${framesPath}`);
  console.log(`training_pairs=${trainingPairsPath}`);
  console.log(`manifest=${manifestPath}`);
}

try {
  main();
} catch (error) {
  console.error(`build-cosyworld-kernel-corpus: ${error.message}`);
  process.exit(1);
}
