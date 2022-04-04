"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.throwUnknownDataType = exports.getType = void 0;
function getType(o) {
    return Object.prototype.toString.call(o);
}
exports.getType = getType;
function throwUnknownDataType(o) {
    throw new TypeError('unsupported data type: ' + getType(o));
}
exports.throwUnknownDataType = throwUnknownDataType;
